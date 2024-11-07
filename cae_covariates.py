import os, gc, shutil, random, joblib, logging
import numpy as np
import pandas as pd
import nibabel as nib
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataGenerator(Sequence):
    
    '''Custom data generator to handle 4D batches.'''
    def __init__(self, batch_size, mode, shuffle=True):
        super(DataGenerator, self).__init__()
        self.base_dir = os.getcwd()
        self.input_data_dir = os.path.join(self.base_dir, 'data')
        self.train_data_dir = os.path.join(self.input_data_dir, 'training')
        self.test_data_dir = os.path.join(self.input_data_dir, 'testing')
        self.val_data_dir = os.path.join(self.input_data_dir, 'validation')
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
        elif self.mode == 'validation':
            imgs = os.listdir(self.val_data_dir)
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


class EncodeT1Data:
    """
    Run ANTs CAE model trained on ADNI dataset
    to encode ADNI data into lower-dimensional latent represtentation
    for covariate analysis
    """

    def __init__(self, batch_size, cae_model_path):
        self.input_shape = (180, 220, 180, 1)
        self.batch_size = batch_size
        self.cae_model_path = cae_model_path
        self.data = ABPosDataGenerator(batch_size=self.batch_size)
        self.ds = self.get_dataset(self.data)
        self.encoded_data = os.path.join(os.getcwd(), 'data', 'T1BrainMNIdims', 'apos_tau_vector_encoding.csv')
        self.encoded_data_dir = os.path.join(os.getcwd(), 'data', 'T1BrainMNIdims', 'apos_encoded_data')
        # mixed precision for training trades computation time for memory
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

    def get_dataset(self, data_generator, repeat=False):
        dataset = tf.data.Dataset.from_generator(
                lambda: data_generator_wrapper(data_generator),
                output_types=(tf.float32, tf.float32),
                output_shapes=(tf.TensorShape([None,180,220,180,1]), tf.TensorShape([None,128])))
        if repeat:
            dataset = dataset.repeat(data_generator.__len__()).batch(self.batch_size)
        else:
            dataset = dataset.batch(self.batch_size)
        return dataset
    
    def get_cae(self):
        cae_model = load_model(self.cae_model_path, compile=True)
        return cae_model
    
    def get_cae_encoder(self):
        cae_model = self.get_cae()
        encoder_output = cae_model.get_layer(name='dense').output
        encoder = Model(inputs=cae_model.input, outputs=encoder_output)
        encoder.compile()
        return encoder

    def encode_images(self):
        df = pd.DataFrame(columns=['ImgCode','VectorEncoding'])
        encoder = self.get_cae_encoder()
        for i in range(self.data.__len__()):
            x, nms = self.data[i]
            encoded_imgs = encoder.predict(x)
            for idx, fn in enumerate(nms):
                n = fn.replace('_T12MNI.nii.gz','')
                e = list(encoded_imgs[idx])
                df.loc[len(df.index)] = [n, e]
        df.to_csv(self.encoded_data, index=False)

    def save_model_to_file(self, model, filepath):
        save_model(model, filepath)
    
    def get_slices(self, image):
        depth = image.shape[1]
        idx_1_3 = depth//3
        idx_1_2 = depth//2
        idx_2_3 = 2 * depth//3
        return [image[:,idx_1_3,:], image[:,idx_1_2,:], image[:,idx_2_3,:]]
    
    def plot_orig_and_recon(self, n_samples, filepath):
        # get sample of reconstructed images
        cae = self.get_cae()
        orig_images = self.data.get_random_sample(n_samples)
        recon_images = cae.predict(orig_images)
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


if __name__=='__main__':
    gc.collect()
    tf.keras.backend.clear_session()
    path = '/m/Researchers/Eliana/T1VariationalAutoencoders/saved_models/CAE_all/cae_fmap10_alldata.keras'
    tau_encoder = EncodeABPosTauData(batch_size=5, cae_model_path=path)
    encoded_data = tau_encoder.encode_images()


