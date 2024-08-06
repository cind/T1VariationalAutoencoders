import math
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Conv3D, Conv3DTranspose, Dense, Flatten, Reshape
from tensorflow.keras.optimizers import Adam, schedules

# Probabalistic variational autoencoder module using tensorflow-probability


class ProbVAE(Model):
    """
    Variational autoencoder model
    """
    def __init__(self, input_shape, fmap_size, kl_weight, **kwargs):
        super(ProbVAE, self).__init__(**kwargs)
        self.input_shape = input_shape
        self.fmap_size = fmap_size
        self.kl_weight = tf.Variable(kl_weight, trainable=False, dtype=tf.float16)
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        # architecture parameters
        self.strides = (2, 2, 2)
        self.kernel = (5, 5, 5)
        self.activation = 'relu'
        self.filters = (32, 64, 128)
        self.factor = 2 ** len(self.filters)
        # build encoder and decoder components
        self.encoder = self.create_variational_encoder()
        self.decoder = self.create_variational_decoder()
        # define optimizer
        lr_sched = schedules.ExponentialDecay(initial_learning_rate=0.0001, decay_steps=1000, decay_rate=0.9)
        self.opt = Adam(learning_rate=lr_sched, clipnorm=1.0)

    def reparameterize(self, mean, logvar):
        batch = len(mean)
        latent = len(mean[0])
        eps = tf.random.normal(shape=(batch, latent))
        eps = tf.cast(eps, dtype=tf.float16)
        return eps * tf.exp(0.5 * logvar) + mean

    def encode(self, inputs):
        tf.print('input shape:', inputs.shape)
        mean, logvar = self.encoder(inputs)
        return mean, logvar

    def decode(self, z, apply_sigmoid=False):
        recon = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(recon)
            return probs
        return recon
    
    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        sample = tf.cast(sample, dtype=tf.float16)
        mean = tf.cast(mean, dtype=tf.float16)
        logvar = tf.cast(logvar, dtype=tf.float16)
        log2pi = tf.math.log(2.0*np.pi)
        log2pi = tf.cast(log2pi, dtype=tf.float16)
        sq_diff = (sample - mean) ** 2
        sq_diff = tf.cast(sq_diff, dtype=tf.float16)
        elogvar = tf.exp(-logvar)
        elogvar = tf.cast(elogvar, dtype=tf.float16)
        rslt = tf.reduce_sum(-0.5 * (sq_diff * elogvar + logvar + log2pi), axis=raxis)
        return rslt
    
    def compute_loss(self, inputs):
        """
        Forward pass through model and return total loss.
        Computed by optimizing single-sample Monte Carlo estimate of ELBO.
        log(p(x|z)): reconstruction loss (MSE)
        log(p(z)): prior distribution on latent space
        log(q(z|x)): encoder distribution
        """
        inputs = tf.cast(inputs, dtype=tf.float16)
        mean, logvar = self.encode(inputs)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        logpx_z = tf.reduce_mean(tf.square(inputs - recon))
        logpz = self.log_normal_pdf(z, 0.0, 0.0)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        loss = -tf.reduce_mean(logpx_z + logpz - logqz_x)
        self.track(loss)
        return loss
    
    def call(self, inputs):
        """
        Defines model forward pass.
        Takes in data and returns reconstructed data.
        """
        mean, logvar = self.encode(inputs)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon
    
    def build(self):
        self.compile(optimizer=self.opt, metrics=self.metrics)
    
    @tf.function
    def train_step(self, inputs):
        """
        Executes one training step and returns loss.
        """
        with tf.GradientTape() as tape:
            loss = self.compute_loss(inputs)
        grads = tape.gradient(loss, self.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.trainable_weights))
        return loss
    
    @tf.function
    def test_step(self, inputs):
        loss = self.compute_loss(inputs)
        return loss
    
    def create_variational_encoder(self):
        padding = 'valid'
        if self.input_shape[0] % self.factor == 0:
            padding = 'same'
        inputs = Input(shape = self.input_shape)
        x = inputs
        
        for i in range(len(self.filters)):
            local_padding = 'same'
            kernel_size = self.kernel
            if i == (len(self.filters) - 1):
                local_padding = padding
                kernel_size = tuple(np.array(self.kernel) - 2)

            x = Conv3D(filters=self.filters[i],
                         kernel_size=kernel_size,
                         strides=self.strides,
                         activation=self.activation,
                         padding=local_padding,
                         kernel_initializer='he_normal')(x)

        x = Flatten()(x)
        z_mean = Dense(self.fmap_size, name='z_mean')(x)
        z_log_var = Dense(self.fmap_size, name='z_log_var')(x)
        
        encoder = Model(inputs, [z_mean, z_log_var], name='encoder')
        return encoder

    def create_variational_decoder(self):
        padding = 'valid'
        if self.input_shape[0] % self.factor == 0:
            padding = 'same'
        shape_before_flatten = (self.input_shape[0]//self.factor,
                                self.input_shape[1]//self.factor,
                                self.input_shape[2]//self.factor,
                                self.filters[-1])
        decoder_input = Input(shape=(self.fmap_size,))
        x = Dense(np.prod(shape_before_flatten), activation=self.activation)(decoder_input)
        x = Reshape(target_shape=shape_before_flatten)(x)
        
        for i in range(len(self.filters), 1, -1):
            local_padding = 'same'
            kernel_size = self.kernel
            if i == len(self.filters):
                local_padding = padding
                kernel_size = tuple(np.array(self.kernel) - 2)

            x = Conv3DTranspose(filters=self.filters[i-1],
                kernel_size=kernel_size,
                strides=self.strides,
                activation=self.activation,
                padding=local_padding,
                kernel_initializer='he_normal')(x) 

        outputs = Conv3DTranspose(filters=self.input_shape[-1],
            kernel_size=self.kernel,
            strides=self.strides,
            padding='same',
            kernel_initializer='he_normal')(x)

        decoder = Model(decoder_input, outputs, name='decoder')
        return decoder

    @property
    def metrics(self):
        return [self.total_loss_tracker]
    
    def track(self, loss):
        self.total_loss_tracker.update_state(loss)
    
    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.fmap_size))
        return self.decode(eps, apply_sigmoid=True)
    
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

