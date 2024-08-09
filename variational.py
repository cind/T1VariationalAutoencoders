import math
import gc
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import ops
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Conv3D, Conv3DTranspose, Dense, Flatten, Reshape
from tensorflow.keras.losses import Loss, MeanSquaredError
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras.callbacks import Callback


class VariationalLoss(Loss):
    """
    Computes loss for VAE as
    total_loss = MSE + kl_weight * kl_loss
    """
    def __init__(self, kl_weight, name='variational_loss'):
        super().__init__(name=name)
        self.recon_loss_fn = MeanSquaredError()
        self.kl_weight = kl_weight

    def __call__(self, inputs, recon, z_mean, z_log_var):
        recon_loss = tf.cast(self.recon_loss_fn(inputs, recon), dtype=tf.float16)
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        total_loss = recon_loss + self.kl_weight * kl_loss
        return total_loss, recon_loss, kl_loss


class Sampling(Layer):
    """
    Inputs: z_mean, z_log_var --> encoder outputs
    Outputs: resampled z --> decoder inputs
    Uses (z_mean, z_log_var) to sample the latent space and generate new z.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(666)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch,dim), seed=self.seed_generator)
        epsilon = tf.cast(epsilon, dtype=tf.float16)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class KLAnnealing(Callback):
    """
    Applies KL divergence annealing to gradually increase KL weight over several epochs.
    """
    def __init__(self, vae, kl_start, kl_end, annealing_epochs):
        super(KLAnnealing, self).__init__()
        self.vae = vae
        self.kl_start = kl_start
        self.kl_end = kl_end
        self.annealing_epochs = annealing_epochs
        self.kl_schedule = np.linspace(kl_start, kl_end, annealing_epochs)

    def on_epoch_begin(self, epoch, logs=None): 
        if epoch < self.annealing_epochs:
            new_kl_weight = self.kl_schedule[epoch]
        else:
            new_kl_weight = self.kl_end
        self.vae.kl_weight.assign(new_kl_weight)
        print(f"\nEpoch {epoch+1}: KL weight set to {new_kl_weight}")


class VAE(Model):
    """
    Variational autoencoder model
    """
    def __init__(self, input_shape, fmap_size, kl_weight, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.input_shape = input_shape
        self.fmap_size = fmap_size
        self.kl_weight = tf.Variable(kl_weight, trainable=False, dtype=tf.float16)
        # set architecture parameters
        self.activation = 'relu'
        self.strides = (2,2,2)
        self.filters = [32,64,128]
        self.factor = 2 ** len(self.filters)
        self.encoder = self.create_variational_encoder()
        self.decoder = self.create_variational_decoder()
        # set loss, metrics, optimizer
        self.loss_fn = VariationalLoss(kl_weight=self.kl_weight)
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.recon_loss_tracker = keras.metrics.Mean(name='recon_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')
        lr_sched = schedules.ExponentialDecay(initial_learning_rate=0.0001, decay_steps=1000, decay_rate=0.9)
        optimizer = Adam(learning_rate=lr_sched, clipnorm=1.0)
        self.opt = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]
    
    def build(self):
        self.compile(loss=self.loss_fn, optimizer=self.opt, metrics=self.metrics)
    
    def call(self, inputs):
        """
        Defines forward pass
        Inputs: image inputs
        Outputs: reconstructed image, z_mean, z_log_var (needed for loss computation)
        """
        z_mean, z_log_var, z = self.encoder(inputs)
        recon = self.decoder(z)
        return recon
    
    def track_metrics(self, total_loss, recon_loss, kl_loss):
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
    
    @tf.function
    def train_step(self, inputs):
        inputs = inputs[0]
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
        z_mean, z_log_var, z = self.encoder(inputs)
        recon = self.decoder(z)
        total_loss, recon_loss, kl_loss = self.loss_fn(inputs, recon, z_mean, z_log_var)
        self.track_metrics(total_loss, recon_loss, kl_loss)
        return {m.name: m.result() for m in self.metrics}

    def create_variational_encoder(self):
        inputs = Input(shape = self.input_shape)
        x = inputs
        x = Conv3D(filters=self.filters[0], kernel_size=(5,5,5), strides=self.strides, 
                activation=self.activation, padding='same', kernel_initializer='he_normal')(x)
        x = Conv3D(filters=self.filters[1], kernel_size=(5,5,5), strides=self.strides,
                activation=self.activation, padding='same', kernel_initializer='he_normal')(x)
        x = Conv3D(filters=self.filters[2], kernel_size=(3,3,3), strides=self.strides,
                activation=self.activation, padding='valid', kernel_initializer='he_normal')(x)
        x = Flatten()(x)
        z_mean = Dense(self.fmap_size, name='z_mean')(x)
        z_log_var = Dense(self.fmap_size, name='z_log_var')(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        return encoder

    def create_variational_decoder(self):
        shape_before_flatten = (self.input_shape[0]//self.factor,
                                self.input_shape[1]//self.factor,
                                self.input_shape[2]//self.factor,
                                self.filters[-1])
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
                       'kl_weight': self.kl_weight.numpy()})
        return config

    @classmethod
    def from_config(cls, config):
        input_shape = config.pop('input_shape')
        fmap_size = config.pop('fmap_size')
        kl_weight = config.pop('kl_weight')
        return cls(input_shape=input_shape, fmap_size=fmap_size, kl_weight=kl_weight, **config)





