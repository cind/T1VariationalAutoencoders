import math
import gc
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Conv3D, Conv3DTranspose, Dense, Flatten, Reshape
from tensorflow.keras.optimizers import Adam, schedules

# Probabalistic variational autoencoder module using tensorflow-probability
tfd = tfp.distributions
tfpl = tfp.layers


class ProbVAE(Model):
    """
    Variational autoencoder model
    """
    def __init__(self, input_shape, fmap_size, kl_weight, batch_size, **kwargs):
        super(ProbVAE, self).__init__(**kwargs)
        self.input_shape = input_shape
        self.fmap_size = fmap_size
        self.batch_size = batch_size
        self.kl_weight = tf.Variable(kl_weight, trainable=False, dtype=tf.float32)
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        # architecture parameters
        self.strides = (2, 2, 2)
        self.activation = 'relu'
        self.filters = [32, 64, 128]
        self.factor = 2 ** len(self.filters)
        # build encoder and decoder components
        self.encoder = self.create_variational_encoder()
        self.decoder = self.create_variational_decoder()
        # define optimizer
        lr_sched = schedules.ExponentialDecay(initial_learning_rate=0.0001, decay_steps=1000, decay_rate=0.9)
        optimizer = Adam(learning_rate=lr_sched, clipnorm=1.0)
        self.opt = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    def reparameterize(self, mean, logvar):
        batchsize = keras.ops.shape(mean)[0]
        eps = tf.random.normal(shape=(batchsize, self.fmap_size))
        eps = tf.cast(eps, dtype=tf.float16)
        return eps * tf.exp(0.5 * logvar) + mean

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.cast(tf.math.log(2.0*np.pi), dtype=tf.float16)
        sq_diff = (sample - mean) ** 2
        elogvar = tf.exp(-logvar)
        elogvar = tf.cast(elogvar, dtype=tf.float16)
        rslt = tf.reduce_sum(-0.5 * (sq_diff * elogvar + logvar + log2pi), axis=raxis)
        return rslt
    
    def compute_loss(self, data, recon, z, z_mean, z_logvar):
        """
        Called after forward pass through model and returns total loss.
        Computed by optimizing single-sample Monte Carlo estimate of ELBO.
        log(p(x|z)): reconstruction loss (MSE)
        log(p(z)): prior distribution on latent space
        log(q(z|x)): encoder distribution
        """
        data = tf.cast(data, dtype=tf.float16)
        logpx_z = tf.reduce_mean(tf.square(data - recon))
        logpz = self.log_normal_pdf(z, 0.0, 0.0)
        logqz_x = self.log_normal_pdf(z, z_mean, z_logvar)
        loss = -tf.reduce_mean(logpx_z + logpz - logqz_x)
        self.track(loss)
        return loss
    
    def call(self, inputs):
        """
        Defines model forward pass.
        Takes in data and returns reconstructed data.
        """
        inputs = tf.cast(inputs, dtype=tf.float32)
        z_mean, z_logvar = self.encoder(inputs)
        z = self.reparameterize(z_mean, z_logvar)
        recon = self.decoder(z)
        self.add_loss(self.compute_loss(inputs, recon, z, z_mean, z_logvar))
        return recon
    
    def forward_pass(self, inputs):
        """
        Wrapper for forward pass computation in train/test step.
        Ensures datatype consistency and returns intermediate values: 
        z, z_mean, z_logvar as well as reconstruction for loss computation.
        """
        inputs = tf.cast(inputs, dtype=tf.float32)
        z_mean, z_logvar = self.encoder(inputs)
        #z_mean = tf.cast(z_mean, dtype=tf.float32)
        #z_logvar = tf.cast(z_logvar, dtype=tf.float32)
        z = self.reparameterize(z_mean, z_logvar)
        #z = tf.cast(z, dtype=tf.float32)
        recon = self.decoder(z)
        #recon = tf.cast(recon, dtype=tf.float32)
        return recon, z, z_mean, z_logvar
    
    def build(self):
        self.compile(optimizer=self.opt, metrics=self.metrics)
    
    @tf.function
    def train_step(self, inputs):
        """
        Executes one training step and returns loss.
        """
        inputs = inputs[0]
        with tf.GradientTape() as tape:
            recon, z, z_mean, z_logvar = self.forward_pass(inputs)
            loss = self.compute_loss(inputs, recon, z, z_mean, z_logvar)
        grads = tape.gradient(loss, self.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.trainable_weights))
        return loss
    
    @tf.function
    def test_step(self, inputs):
        inputs = inputs[0]
        recon, z, z_mean, z_logvar = self.forward_pass(inputs)
        loss = self.compute_loss(inputs, recon, z, z_mean, z_logvar)
        return loss
    
    def create_encoder(self):
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
        encoder = Model(inputs, [z_mean, z_log_var], name='encoder')
        return encoder

    def create_decoder(self):
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

    #def create_
    
    @property
    def metrics(self):
        return [self.total_loss_tracker]
    
    def track(self, loss):
        self.total_loss_tracker.update_state(loss)
    
    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.fmap_size))
        return self.decoder(eps)
    
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

