import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import ops
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Conv3D, Conv3DTranspose, Dense, Flatten, Reshape
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import Callback


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the latent space vector encoding an image."""
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


def create_variational_encoder_decoder(input_image_size,
                                       number_of_filters_per_layer=(32, 64, 128, 10),
                                       convolution_kernel_size=(5, 5, 5),
                                       deconvolution_kernel_size=(5, 5, 5)
                                      ):
    """Function for creating the encoder and decoder components
    of a 3-D symmetric variational convolutional autoencoder model."""

    activation = 'relu'
    strides = (2, 2, 2)
    latent_dim = number_of_filters_per_layer[-1]
    number_of_encoding_layers = len(number_of_filters_per_layer) - 1
    factor = 2 ** number_of_encoding_layers

    padding = 'valid'
    if input_image_size[0] % factor == 0:
        padding = 'same'

    inputs = Input(shape = input_image_size)
    x = inputs

    # Encoder
    for i in range(number_of_encoding_layers):
        local_padding = 'same'
        kernel_size = convolution_kernel_size
        if i == (number_of_encoding_layers - 1):
            local_padding = padding
            kernel_size = tuple(np.array(convolution_kernel_size) - 2)

        x = Conv3D(filters=number_of_filters_per_layer[i],
                         kernel_size=kernel_size,
                         strides=strides,
                         activation=activation,
                         padding=local_padding,
                         kernel_initializer='he_normal')(x)

    shape_before_flattening = tf.keras.backend.int_shape(x)[1:]
    x = Flatten()(x)
    #x = Dense(units=256, activation=activation)(x)

    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])

    # Decoder
    decoder_input = Input(shape=(latent_dim,))
    x = Dense(np.prod(shape_before_flattening), activation=activation)(decoder_input)
    x = Reshape(target_shape=shape_before_flattening)(x)

    for i in range(number_of_encoding_layers, 1, -1):
        local_padding = 'same'
        kernel_size = convolution_kernel_size
        if i == number_of_encoding_layers:
            local_padding = padding
            kernel_size = tuple(np.array(deconvolution_kernel_size) - 2)

        x = Conv3DTranspose(filters=number_of_filters_per_layer[i-1],
                kernel_size=kernel_size,
                strides=strides,
                activation=activation,
                padding=local_padding,
                kernel_initializer='he_normal')(x) 
    
    outputs = Conv3DTranspose(filters=input_image_size[-1],
            kernel_size=deconvolution_kernel_size,
            strides=strides,
            padding='same',
            kernel_initializer='he_normal')(x)

    # model outputs
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    decoder = Model(decoder_input, outputs, name='decoder')
    return encoder, decoder


class KLAnnealing(Callback):
    """Applies KL divergence annealing to gradually increase KL weight over several epochs."""
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
    """Handles forward pass and computes loss."""
    def __init__(self, encoder, decoder, kl_weight, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = tf.Variable(kl_weight, trainable=False, dtype=tf.float16)
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.recon_loss_tracker = keras.metrics.Mean(name='recon_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        data = tf.cast(data, dtype=tf.float16)
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            recon_loss = tf.reduce_mean(tf.square(data - reconstruction))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            total_loss = recon_loss + self.kl_weight * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {'loss': self.total_loss_tracker.result(),
                'recon_loss': self.recon_loss_tracker.result(),
                'kl_loss': self.kl_loss_tracker.result()}




