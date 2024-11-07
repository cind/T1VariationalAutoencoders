import os, math, gc, logging, random
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow import keras
from keras import ops
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Conv3D, Conv3DTranspose, Dense, Flatten, Reshape, Dropout, SpatialDropout3D
from tensorflow.keras.losses import Loss, MeanSquaredError
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import register_keras_serializable, Progbar

logger = logging.getLogger(__name__)

# structural similarity index and peak signal to noise ratio as additional reconstruction measures

def compute_ssim(img1, img2):
    return tf.reduce_mean(tf.image.ssim(img1, img2, max_val=1.0))

def compute_psnr(img1, img2):
    return tf.reduce_mean(tf.image.psnr(img1, img2, max_val=1.0))


@register_keras_serializable(package='variational')
class VariationalLoss(Loss):
    """
    Computes loss for VAE as total_loss = MSE + kl_weight * kl_loss
    Additional gradient penalty regularization and weight map for foreground focus
    """
    def __init__(self, kl_weight, name='variational_loss', reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE):
        super().__init__(name=name, reduction=reduction)
        self.mse = MeanSquaredError()
        self.kl_weight = kl_weight

    def get_config(self):
        config = super(VariationalLoss, self).get_config()
        return config
    
    def __call__(self, inputs, recon, z_mean, z_log_var, weight_map=None):
        recon_weight = 100
        #ssim_weight = 0.5
        #psnr_weight = 0.1
        inputs = tf.cast(inputs, dtype=tf.float16)
        recon_loss = tf.cast(tf.reduce_mean(self.mse(inputs, recon)), dtype=tf.float16)
        if weight_map:
            recon_loss = weight_map * recon_loss
        # add gradient penalty
        #gradient_penalty = self.gradient_penalty_loss(inputs, recon)
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        kl_loss = tf.cast(tf.reduce_mean(kl_loss), dtype=tf.float16)
        #ssim_loss = -tf.cast(compute_ssim(inputs, recon), dtype=tf.float16)
        #psnr_loss = -tf.cast(compute_psnr(inputs, recon), dtype=tf.float16)
        total_loss = recon_weight*recon_loss + self.kl_weight*kl_loss
        return total_loss, recon_loss, kl_loss

    def gradient_penalty_loss(self, y_true, y_pred, sample_weight=None):
        """Gradient penalty regularization"""
        gradients = tf.gradients(y_pred, y_true)[0]
        gradient_l2_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2,3]))
        gradient_penalty = tf.reduce_mean(tf.square(gradient_l2_norm - 1.0))
        gradient_penalty = tf.cast(gradient_penalty, dtype=tf.float16)
        return gradient_penalty
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Sampling(Layer):
    """
    Inputs: z_mean, z_log_var --> encoder outputs
    Outputs: resampled z --> decoder inputs
    Uses (z_mean, z_log_var) to sample the latent space and generate new z.
    """
    def __init__(self, regularizer=False, **kwargs):
        super(Sampling, self).__init__(**kwargs)
        self.regularizer = regularizer
        self.seed_generator = keras.random.SeedGenerator(1729)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch,dim), seed=self.seed_generator)
        epsilon = tf.cast(epsilon**2, dtype=z_mean.dtype)
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return z


class KLAnnealing(Callback):
    """
    Applies KL divergence annealing to gradually increase KL weight linearly over several epochs.
    """
    def __init__(self, vae, validation_data, kl_start, kl_end, annealing_epochs, start_epoch=0, verbose=1):
        super(KLAnnealing, self).__init__()
        self.vae = vae
        self.validation_data = validation_data
        self.kl_start = kl_start
        self.kl_end = kl_end
        self.annealing_epochs = annealing_epochs
        self.start_epoch = start_epoch
        self.verbose = verbose
        self.kl_increment = 0.1
        self.kl_schedule = np.linspace(kl_start, kl_end, annealing_epochs)

    def on_epoch_begin(self, epoch, logs=None): 
        if epoch >= self.start_epoch and epoch < self.start_epoch + self.annealing_epochs:
            new_kl_weight = self.kl_schedule[epoch - self.start_epoch]
        elif epoch >= self.start_epoch + self.annealing_epochs:
            new_kl_weight = self.kl_end
        else:
            new_kl_weight = self.kl_start
        if hasattr(self.vae, 'kl_weight'):
            tf.keras.backend.set_value(self.vae.kl_weight, new_kl_weight)
        #self.vae.kl_weight.assign(new_kl_weight)
        print(f"\nEpoch {epoch+1}: KL weight set to {new_kl_weight}.")

    def on_epoch_end(self, epoch, logs=None):
        pass


class LatentSpaceVarMonitoring(Callback):
    """
    If variance drops below threshold, increase KL weight
    """
    def __init__(self, vae, validation_data, var_threshold, kl_increment, start_epoch, verbose=1):
        super(LatentSpaceVarMonitoring, self).__init__()
        self.vae = vae
        self.validation_data = validation_data
        self.var_threshold = var_threshold
        self.kl_increment = kl_increment
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start_epoch:
            idx = np.random.randint(len(self.validation_data[0]))
            z_mean, z_log_var, z = self.vae.encoder.predict(np.expand_dims(self.validation_data[0][idx], axis=0))
            mean_var = np.mean(z_log_var)
            logger.info(f'\nEpoch {epoch+1}: Mean variance in latent space: {mean_var}')
            if mean_var < self.var_threshold:
                new_kl_weight = self.vae.kl_weight + self.kl_increment
                self.vae.kl_weight.assign(new_kl_weight)
                logger.info(f'KL weight increased to {new_kl_weight}')


class VAE(Model):
    """
    Variational autoencoder model
    """
    def __init__(self, input_shape, fmap_size, kl_weight, batch_size, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.input_shape = input_shape
        self.fmap_size = fmap_size
        self.kl_weight = tf.Variable(kl_weight, trainable=False, dtype=tf.float16)
        self.batch_size = batch_size
        self.history = None
        self.mask_file = os.path.join(os.getcwd(), 'atlases', 'MNI152_T1_1mm_mask.nii.gz')
        self.ds_mask_file = os.path.join(os.getcwd(), 'atlases', 'ds_mask.nii.gz')
        # set architecture parameters
        self.activation = 'relu'
        self.strides = (2,2,2)
        self.filters = [32,64,128]
        self.factor = 2 ** len(self.filters)
        self.lr = 0.001
        self.encoder = self.create_variational_encoder()
        self.decoder = self.create_variational_decoder()
        # set loss, metrics, optimizer
        self.loss_fn = VariationalLoss(kl_weight=self.kl_weight)
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.recon_loss_tracker = keras.metrics.Mean(name='MSE_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='KL_loss')
        self.grad_penalty_tracker = keras.metrics.Mean(name='gradient_penalty_loss')
        self.ssim_loss_tracker = keras.metrics.Mean(name='SSIM_loss')
        self.psnr_loss_tracker = keras.metrics.Mean(name='PSNR_loss')
        lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.lr, decay_steps=1000, decay_rate=0.9)
        opt0 = tf.keras.optimizers.Adam(learning_rate=lr_sched, clipnorm=1.0)
        opt1 = tf.keras.optimizers.Adam(learning_rate=lr_sched)
        self.opt = tf.keras.mixed_precision.LossScaleOptimizer(opt0)

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker, 
                self.ssim_loss_tracker, self.psnr_loss_tracker]
    
    def get_mask(self):
        """Load brain mask and convert to tensor"""
        mask = nib.load(self.mask_file).get_fdata()
        mask = np.pad(mask, pad_width=((0,0),(1,1),(0,0)), mode='constant', constant_values=0)
        mask = mask[1:181,:,1:181]
        mask = tf.constant(mask, dtype=tf.float16)
        return mask
    
    def get_weight_map(self, fg_weight=5.0, bg_weight=1):
        """Create weight map from brain mask"""
        mask = self.get_mask()
        weight_map = tf.where(tf.equal(mask, 1), fg_weight, bg_weight)
        weight_map = tf.cast(weight_map, dtype=tf.float16)
        return weight_map
    
    def expand_feature_dim(self, array, match_fmap=False):
        if match_fmap:
            exp_array = tf.tile(tf.expand_dims(array, axis=-1), [1,1,1,self.fmap_size])
        else:
            exp_array = tf.expand_dims(array, axis=-1)
        return exp_array
    
    def expand_for_batch(self, array, expand_feature_dim):
        if expand_feature_dim:
            array = self.expand_feature_dim(array, match_fmap=True)
        exp_array = tf.tile(tf.expand_dims(array, axis=0), [self.batch_size,1,1,1,1])
        exp_array = tf.cast(exp_array, dtype=tf.float16)
        return exp_array
    
    def get_shape_before_flatten(self, include_feature_dim):
        if include_feature_dim:
            return (self.input_shape[0]//self.factor,
                    self.input_shape[1]//self.factor,
                    self.input_shape[2]//self.factor,
                    self.filters[-1])
        else:
            return (self.input_shape[0]//self.factor,
                    self.input_shape[1]//self.factor,
                    self.input_shape[2]//self.factor)

    def build(self):
        self.compile(loss=self.loss_fn, optimizer=self.opt, metrics=self.metrics)
    
    def call(self, inputs):
        """
        Defines forward pass
        Inputs: image inputs
        Outputs: reconstructed image
        """
        z_mean, z_log_var, z = self.encoder(inputs)
        recon = self.decoder(z)
        return recon
    
    def track_metrics(self, total_loss, recon_loss, kl_loss, gradient_penalty=None, ssim_loss=None, psnr_loss=None):
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        if gradient_penalty is not None:
            self.grad_penalty_tracker.update_state(gradient_penalty)
        if ssim_loss is not None:
            self.ssim_loss_tracker.update_state(ssim_loss)
        if psnr_loss is not None:
            self.psnr_loss_tracker.update_state(psnr_loss)
    
    @tf.function
    def train_step(self, inputs):
        inputs = inputs[0]
        weight_map = self.get_weight_map()
        exp_weight_map = self.expand_for_batch(weight_map, expand_feature_dim=True)
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(inputs)
            recon = self.decoder(z)
            total_loss, recon_loss, kl_loss = self.loss_fn(inputs, recon, z_mean, z_log_var)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.trainable_weights))
        self.track_metrics(total_loss, recon_loss, kl_loss)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, inputs):
        inputs = inputs[0]
        weight_map = self.get_weight_map()
        exp_weight_map = self.expand_for_batch(weight_map, expand_feature_dim=True)
        z_mean, z_log_var, z = self.encoder(inputs)
        recon = self.decoder(z)
        total_loss, recon_loss, kl_loss = self.loss_fn(inputs, recon, z_mean, z_log_var)
        self.track_metrics(total_loss, recon_loss, kl_loss)
        return {m.name: m.result() for m in self.metrics}

    def fit(self, *args, **kwargs):
        """Set attributes in keras fit method"""
        self.verbose = kwargs.get('verbose', 1)
        self.history = super(VAE, self).fit(*args, **kwargs)
        return self.history
    
    def custom_fit(self, model, train_data, val_data, epochs, callbacks, verbose=1):
        """Add progress bar to custom training loop"""
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            progbar = Progbar(target=len(train_data), stateful_metrics=['loss'])
            for step, x_batch_train in enumerate(train_data):
                result = model.train_step(x_batch_train)
                progbar.update(step+1, values=[('loss', result['loss'])])
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs=result)
            val_progbar = Progbar(target=len(val_data), stateful_metrics=['val_loss']) 
            for val_step, x_batch_val in enumerate(val_data):
                val_result = model.test_step(x_batch_val)
                val_progbar.update(val_step+1, values=['val_loss', val_result['loss']])
            print(f'Training loss: {result["loss"]:.4f}')
            print(f'Validation loss: {val_result["loss"]:.4f}')
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs=val_result)
    
    def create_variational_encoder(self):
        inputs = Input(shape = self.input_shape)
        x = inputs
        x = Conv3D(filters=self.filters[0], kernel_size=(5,5,5), strides=self.strides, 
                activation=self.activation, padding='same', kernel_initializer='he_normal')(x)
        x = Conv3D(filters=self.filters[1], kernel_size=(5,5,5), strides=self.strides,
                activation=self.activation, padding='same', kernel_initializer='he_normal')(x)
        x = Conv3D(filters=self.filters[2], kernel_size=(3,3,3), strides=self.strides,
                activation=self.activation, padding='valid', kernel_initializer='he_normal')(x)
        #x = SpatialDropout3D(0.2, data_format='channels_last')(x)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        z_mean = Dense(self.fmap_size, name='z_mean', kernel_initializer='he_normal')(x)
        z_log_var = Dense(self.fmap_size, name='z_log_var', kernel_initializer='he_normal')(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        return encoder

    def create_variational_decoder(self):
        shape_before_flatten = self.get_shape_before_flatten(include_feature_dim=True)
        decoder_input = Input(shape=(self.fmap_size,))
        x = Dense(np.prod(shape_before_flatten), activation=self.activation)(decoder_input)
        x = Reshape(target_shape=shape_before_flatten)(x)
        x = Conv3DTranspose(filters=self.filters[1], kernel_size=(3,3,3), strides=self.strides,
                activation=self.activation, padding='valid', kernel_initializer='he_normal')(x)
        x = Conv3DTranspose(filters=self.filters[0], kernel_size=(5,5,5), strides=self.strides,
                activation=self.activation, padding='same', kernel_initializer='he_normal')(x)
        outputs = Conv3DTranspose(filters=1, kernel_size=(5,5,5), strides=self.strides,
                activation='linear', padding='same', kernel_initializer='he_normal')(x)
        decoder = Model(decoder_input, outputs, name='decoder')
        return decoder
   
    def get_config(self):
        config = super().get_config().copy()
        config.update({'input_shape': self.input_shape,
                       'fmap_size': self.fmap_size,
                       'kl_weight': self.kl_weight.numpy(),
                       'batch_size': self.batch_size})
        return config

    @classmethod
    def from_config(cls, config):
        input_shape = config.pop('input_shape')
        fmap_size = config.pop('fmap_size')
        kl_weight = config.pop('kl_weight')
        batch_size = config.pop('batch_size')
        return cls(input_shape=input_shape, fmap_size=fmap_size, kl_weight=kl_weight, batch_size=batch_size, **config)





