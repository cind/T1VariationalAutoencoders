import os, sys, shutil, string, csv, subprocess, logging, random, gc
from datetime import datetime
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import nibabel as nib
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.callbacks import EarlyStopping, CallbackList
from antspynet.architectures import create_convolutional_autoencoder_model_3d

# LOCAL IMPORTS
from variational import VariationalLoss, Sampling, BoolMask, VAE, KLAnnealing, LatentSpaceVarMonitoring
from probabalisticVAE import ProbVAE
from utils import exceptions

logger = logging.getLogger(__name__)


class DataGenerator(Sequence):
    
    """
    Custom data generator to handle 4D batches.
    """
    def __init__(self, batch_size, mode, shuffle=True):
        super(DataGenerator, self).__init__()
        self.base_dir = os.getcwd()
        self.input_data_dir = os.path.join(self.base_dir, 'data', 'CN_ABneg')
        self.train_data_dir = os.path.join(self.input_data_dir, 'training')
        self.val_data_dir = os.path.join(self.input_data_dir, 'validation')
        self.test_data_dir = os.path.join(self.input_data_dir, 'testing')
        self.mask_file = os.path.join(self.base_dir, 'atlases', 'MNI152_T1_1mm_mask.nii.gz')
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

    def get_mask(self):
        mask = nib.load(self.mask_file).get_fdata()
        mask = np.pad(mask, pad_width=((0,0),(1,1),(0,0)), mode='constant', constant_values=0)
        mask = mask[1:181,:,1:181]
        return mask
    
    def get_imgs_by_mode(self):
        if self.mode == 'training':
            imgs = os.listdir(self.train_data_dir)
        elif self.mode == 'validation':
            imgs = os.listdir(self.val_data_dir)
        elif self.mode == 'testing':
            imgs = os.listdir(self.test_data_dir)
        return imgs
    
    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index+1) * self.batch_size
        batch_indexes = self.indexes[start_idx:end_idx]
        batch_filenames = [self.filenames[i] for i in batch_indexes]
        x = np.empty((len(batch_indexes), *self.data_shape))
        for i, idx in enumerate(batch_indexes):
            image = os.path.join(self.input_data_dir, self.mode, batch_filenames[i])
            img = nib.load(image)
            data = img.get_fdata()
            # resize from (182,218,182) --> (180,220,180) for network compatibility
            data = np.pad(data, pad_width=((0,0),(1,1),(0,0)), mode='constant', constant_values=0)
            data = data[1:181,:,1:181]
            # min-max scale data from range [0,255] --> [0,1] for training stability
            data = data/255
            # apply mask to explicitly zero out background
            mask = self.get_mask()
            data = data * mask
            x[i,:,:,:,0] = data
        return x

    def get_random_sample(self, n_samples):
        """Use for test data generator only"""
        test_data = self.filenames
        indices = np.random.choice(len(test_data), n_samples, replace=False)
        x = np.empty((n_samples, *self.data_shape))
        for i, idx in enumerate(indices):
            item = test_data[idx]
            item_fullpath = os.path.join(os.getcwd(), 'data', 'CN_ABneg', 'testing', item)
            img = nib.load(item_fullpath)
            data = img.get_fdata() 
            data = np.pad(data, pad_width=((0,0),(1,1),(0,0)), mode='constant', constant_values=0)
            data = data[1:181,:,1:181]
            data = data/255
            mask = self.get_mask()
            data = data * mask
            x[i,:,:,:,0] = data
        return x
    
    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.indexes)


class AutoencoderTransfer():
    """
    AI model using trained ANTs CAE model for transfer learning
    on custom variational autoencoder
    """

    def __init__(self, batch_size, epochs, fmap_size):
        super(AutoencoderTransfer, self).__init__()
        self.input_shape = (180, 220, 180, 1)
        self.batch_size = batch_size
        self.epochs = epochs
        self.fmap_size = fmap_size
        self.train_data = DataGenerator(batch_size=self.batch_size, mode='training')
        self.val_data = DataGenerator(batch_size=self.batch_size, mode='validation')
        self.test_data = DataGenerator(batch_size=self.batch_size, mode='testing')
        # mixed precision for training trades computation time for memory
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

    def load_cae_from_file(self, filepath):
        return load_model(filepath, compile=True)        
    
    def build_vae(self):
        """Builds/compiles VAE"""
        vae = VAE(input_shape=self.input_shape, fmap_size=self.fmap_size, kl_weight=1.0, batch_size=self.batch_size)
        vae.build()
        return vae
     
    def transfer_weights_fromCAE(self, cae, vae):
        """Transfers weights from each CAE layer to each VAE layer where applicable"""
        enc_layers = cae.layers[:5]
        bottleneck_layer = cae.layers[5]
        dec_layers = cae.layers[5:]
        for i in range(len(vae.encoder.layers)):
            if i < len(enc_layers):
                vae.encoder.layers[i].set_weights(enc_layers[i].get_weights())
        #vae.encoder.layers[5].set_weights(bottleneck_layer.get_weights())
        for i in range(len(vae.decoder.layers)):
            if i < len(dec_layers) and i > 0:
                vae.decoder.layers[i].set_weights(dec_layers[i].get_weights())
    
    def train_model(self, model):
        kl_anneal = KLAnnealing(model, self.val_data, 0, 1, 20, 0, verbose=1)
        var_monitor = LatentSpaceVarMonitoring(model, self.val_data, 1e-4, 0.1, 0)
        early_stop = EarlyStopping(monitor='total_loss', verbose=1, patience=5, start_from_epoch=25, mode='min')
        cbs = CallbackList([kl_anneal, early_stop])
        cbs.set_model(model)
        history = model.fit(self.train_data, validation_data=self.val_data, epochs=self.epochs, verbose=1, callbacks=cbs)
        return history

    def test_model(self, model):
        loss = model.evaluate(self.test_data)

    def plot_train_progress(self, train_history):
        pass

    def load_vae_from_file(self, filepath):
        objs = {'VAE': VAE,
                'Sampling': Sampling,
                'BoolMask': BoolMask,
                'VariationalLoss': VariationalLoss,
                'KLAnnealing': KLAnnealing,
                'LatentSpaceVarMonitoring': LatentSpaceVarMonitoring}
        return load_model(filepath, custom_objects=objs, compile=True)
    
    def save_model_to_file(self, model, filepath):
        save_model(model, filepath)
    
    def get_slices(self, image):
        depth = image.shape[2]
        idx_1_3 = depth//3
        idx_1_2 = depth//2
        idx_2_3 = 2 * depth//3
        return [image[:,:,idx_1_3], image[:,:,idx_1_2], image[:,:,idx_2_3]]
    
    def plot_imgs_orig_recon(self, autoencoder, n_samples, filepath):
        # get sample of reconstructed images
        orig_images = self.test_data.get_random_sample(n_samples)
        recon_images = autoencoder.predict(orig_images)
        # plot original and reconstructed side by side
        fig, axes = plt.subplots(n_samples, 6, figsize=(15, n_samples*3))
        slice_lbls = ['1/3', '1/2', '2/3']
        for i in range(n_samples):
            orig_slices = self.get_slices(orig_images[i])
            recon_slices = self.get_slices(recon_images[i])
            # plot original slices
            for j, orig_slice in enumerate(orig_slices):
                axes[i,j].imshow(orig_slice, cmap='gray')
                if i == 0:
                    axes[i,j].set_title(slice_lbls[j])
                axes[i,j].axis('off')    
            # plot reconstructed slices
            for j, recon_slice in enumerate(recon_slices):
                axes[i,j+3].imshow(recon_slice, cmap='gray')
                if i == 0:
                    axes[i,j+3].set_title(slice_lbls[j])
                axes[i,j+3].axis('off')
        # overall titles
        fig.text(0.25, 0.96, 'Original', ha='center', fontsize=16)
        fig.text(0.75, 0.96, 'Reconstructed', ha='center', fontsize=16)
        plt.tight_layout(rect=[0,0,1,0.95])
        plt.savefig(filepath)

    def plot_hists_orig_recon(self, autoencoder, n_samples, filepath, nbins):    
        # get sample of reconstructed images
        orig_images = self.test_data.get_random_sample(n_samples)
        recon_images = autoencoder.predict(orig_images)
        # plot histograms of original and reconstructed side by side
        fig, axes = plt.subplots(n_samples, 2, figsize=(15, n_samples))
        for i in range(n_samples):
            orig_hist, orig_bin_edges = np.histogram(orig_images[i], bins=nbins, range=(0,1))
            axes[i,0].plot(orig_bin_edges[0:-1], orig_hist)
            recon_hist, recon_bin_edges = np.histogram(recon_images[i], bins=nbins, range=(0,1))
            axes[i,1].plot(recon_bin_edges[0:-1], recon_hist)
        # overall titles
        fig.text(0.25, 0.96, 'Original histograms', ha='center', fontsize=16)
        fig.text(0.75, 0.96, 'Reconstructed histograms', ha='center', fontsize=16)
        plt.tight_layout(rect=[0,0,1,0.95])
        plt.savefig(filepath)


if __name__ == '__main__':
    gc.collect()
    tf.keras.backend.clear_session()
    cae_filepath = os.path.join(os.getcwd(), 'saved_models', 'CAE_CN_ABneg', 'fmap128_epochs100.keras')
    vae_filepath = os.path.join(os.getcwd(), 'saved_models', 'transferCAE_VAE', 'fmap128_epochs50_mask.keras')
    viz_filepath = os.path.join(os.getcwd(), 'transfer_vae_fmap128_epochs50_mask.png')
    hist_filepath = os.path.join(os.getcwd(), 'transfer_vae_fmap128_epochs50_mask_hists.png')
    transfer_model = AutoencoderTransfer(batch_size=10, epochs=50, fmap_size=128)
    cae = transfer_model.load_cae_from_file(cae_filepath)
    vae = transfer_model.build_vae()
    transfer_model.transfer_weights_fromCAE(cae, vae)
    print(vae.encoder.summary())
    print(vae.decoder.summary())
    transfer_model.train_model(vae)
    transfer_model.test_model(vae)
    transfer_model.save_model_to_file(vae, filepath=vae_filepath)
    #vae = transfer_model.load_vae_from_file(vae_filepath)
    transfer_model.plot_imgs_orig_recon(vae, n_samples=10, filepath=viz_filepath)
    transfer_model.plot_hists_orig_recon(vae, n_samples=10, filepath=hist_filepath, nbins=128)

