import os, shutil, string, csv, subprocess, logging, random, gc
from datetime import datetime
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import nibabel as nib
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv3D, Conv3DTranspose, Dense, Flatten, Reshape
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras.callbacks import EarlyStopping

# LOCAL IMPORTS
from preproc import BaseT1CAE
from variational_autoencoder import create_variational_encoder_decoder, VAE, KLAnnealing
from utils import exceptions

logger = logging.getLogger(__name__)


class DataGenerator(Sequence):
    
    '''Custom data generator to handle 4D batches.'''
    def __init__(self, batch_size, mode, shuffle=True):
        super(DataGenerator, self).__init__()
        self.base_dir = os.getcwd()
        self.input_data_dir = os.path.join(self.base_dir, 'data', 'CN_ABneg')
        self.train_data_dir = os.path.join(self.input_data_dir, 'training')
        self.test_data_dir = os.path.join(self.input_data_dir, 'testing')
        self.mode = mode
        self.data_shape = (180, 220, 180, 1)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.filenames = self.get_imgs_by_mode()
        self.indexes = list(range(len(self.filenames)))
        self.on_epoch_end()
        if self.shuffle:
            random.shuffle(self.indexes)

    def __len__(self):
        return len(self.filenames)//self.batch_size
 
    def get_imgs_by_mode(self):
        if self.mode == 'training':
            imgs = os.listdir(self.train_data_dir)
        elif self.mode == 'testing':
            imgs = os.listdir(self.test_data_dir)
        return imgs
    
    def __getitem__(self, index):
        # x=y for CAE task
        start_idx = index * self.batch_size
        end_idx = (index+1) * self.batch_size
        batch_indexes = self.indexes[start_idx:end_idx]
        batch_filenames = [self.filenames[i] for i in batch_indexes]
        x = np.empty((len(batch_indexes), *self.data_shape))
        y = np.empty((len(batch_indexes), *self.data_shape))
        for i, idx in enumerate(batch_indexes):
            image = os.path.join(self.input_data_dir, self.mode, batch_filenames[i])
            img = nib.load(image)
            data = img.get_fdata()
            # resize from (182,218,182) --> (180,220,180) for network compatibility
            data = np.pad(data, pad_width=((0,0),(1,1),(0,0)), mode='constant', constant_values=0)
            data = data[1:181,:,1:181]
            # min-max scale data from range [0,255] --> [0,1] for training stability
            data = data/255
            x[i,:,:,:,0] = data
            y[i,:,:,:,0] = data
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.indexes)


class T1VAEModel(BaseT1CAE):

    '''AI model using custom variational autoencoder
    built off ANTs CAE'''

    def __init__(self, batch_size, epochs, fmap_size):
        super(T1VAEModel, self).__init__()
        self.input_shape = (180, 220, 180, 1)
        self.batch_size = batch_size
        self.epochs = epochs
        self.fmap_size = fmap_size
        # mixed precision for training trades computation time for memory
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

    def build_model(self):
        '''Builds/compiles VAE'''
        encoder, decoder = create_variational_encoder_decoder(
                        input_image_size=self.input_shape,
                        number_of_filters_per_layer=(32,64,128,self.fmap_size),
                        convolution_kernel_size=(5,5,5),
                        deconvolution_kernel_size=(5,5,5))
        vae = VAE(encoder, decoder, 0.001)
        lr_schedule = schedules.ExponentialDecay(initial_learning_rate=0.0001, decay_steps=1000, decay_rate=0.9)
        opt = Adam(learning_rate=lr_schedule)
        vae.compile(optimizer=opt)
        return vae, encoder, decoder
    
    def train_model(self, model):
        train_generator = DataGenerator(batch_size=self.batch_size, mode='training')
        kl = KLAnnealing(model, kl_start=0.001, kl_end=1.0, annealing_epochs=10)
        model.fit(train_generator, epochs=self.epochs, callbacks=kl)

    def train_and_save_model(self, model):
        train_generator = DataGenerator(batch_size=self.batch_size, mode='training')
        cb = EarlyStopping(monitor='loss', verbose=1, patience=5, start_from_epoch=15)
        train_history = model.fit(train_generator, epochs=self.epochs, callbacks=cb)
        return train_history
    
    def test_model(self, model):
        test_generator = DataGenerator(batch_size=self.batch_size, mode='testing')
        loss, metric = model.evaluate(test_generator)
        print(f'Testing loss: {loss:.8f}')

    def extract_features(self, model):
        generator = DataGenerator(batch_size=self.batch_size, mode='')
        features = model.predict(generator)

    def plot_train_progress(self, train_history):
        pass

    def save_model_to_file(self, model, filepath):
        save_model(model, filepath)
    
    def get_middle_slice(self, image):
        depth = image.shape[2]
        mid_idx = depth//2
        return image[:,:,mid_idx]
    
    def plot_orig_and_recon(self, autoencoder, n_samples, filepath):
        # get sample of reconstructed images
        test_gen = DataGenerator(batch_size=self.batch_size, mode='testing')
        test_data = test_gen.filenames
        indices = np.random.choice(len(test_data), n_samples, replace=False)
        orig_images = np.empty((n_samples, *test_gen.data_shape))
        for i, idx in enumerate(indices):
            item = test_data[idx]
            item = os.path.join(os.getcwd(), 'data', 'testing', item)
            img = nib.load(item)
            data = img.get_fdata() 
            # resize from (182,218,182) --> (180,220,180) for network compatibility
            data = np.pad(data, pad_width=((0,0),(1,1),(0,0)), mode='constant', constant_values=0)
            data = data[1:181,:,1:181]
            # min-max scale data from range [0,255] --> [0,1] for training stability
            data = data/255
            orig_images[i,:,:,:,0] = data
        recon_images = autoencoder.predict(orig_images)
        # plot original and reconstructed side by side
        fig, axes = plt.subplots(n_samples, 2, figsize=(10, n_samples*3))
        for i in range(n_samples):
            orig_slice = self.get_middle_slice(orig_images[i])
            axes[i,0].imshow(orig_slice, cmap='gray')
            axes[i,0].set_title('Original')
            axes[i,0].axis('off')
            recon_slice = self.get_middle_slice(recon_images[i])
            axes[i,1].imshow(recon_slice, cmap='gray')
            axes[i,1].set_title('Reconstructed')
            axes[i,1].axis('off')
        plt.tight_layout()
        plt.savefig(filepath)


if __name__ == '__main__':
    gc.collect()
    t1vae_model = T1VAEModel(batch_size=13, epochs=100, fmap_size=10)
    vae, encoder, decoder = t1vae_model.build_model()
    print(vae.summary())
    print(encoder.summary())
    print(decoder.summary())
    t1vae_model.train_model(vae)
    t1vae_model.save_model_to_file(vae, filepath=os.path.join(os.getcwd(), 'saved_models', 'vae_fmap10.keras'))
    t1vae_model.test_model(vae)
    t1vae_model.plot_orig_and_recon(vae, n_samples=5, filepath=os.path.join(os.getcwd(), 'img_recon_fmap10.png'))

