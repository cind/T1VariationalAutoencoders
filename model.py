import os, shutil, string, csv, subprocess, logging, random
from datetime import datetime
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import nibabel as nib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dropout, Dense, Concatenate, GlobalAveragePooling3D
from tensorflow.keras.losses import Loss, MeanSquaredError
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from antspynet.architectures import create_convolutional_autoencoder_model_3d

# LOCAL IMPORTS
from preproc import BaseT1CAE
from utils import exceptions

logger = logging.getLogger(__name__)


class DataGenerator(Sequence):
    
    '''Custom data generator to handle 4D batches.'''
    def __init__(self, batch_size, mode, shuffle=True):
        super(DataGenerator, self).__init__()
        self.base_dir = os.getcwd()
        self.input_data_dir = os.path.join(self.base_dir, 'data')
        self.train_data_dir = os.path.join(self.input_data_dir, 'training')
        self.test_data_dir = os.path.join(self.input_data_dir, 'testing')
        self.mode = mode
        self.data_shape = (142, 144, 180, 1)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.filenames, self.labels = self.get_imgs_labels_by_mode()
        self.indexes = list(range(len(self.filenames)))
        self.on_epoch_end()
        if self.shuffle:
            random.shuffle(self.indexes)

    def __len__(self):
        return len(self.filenames)//self.batch_size
 
    def get_imgs_by_mode(self):
        if self.mode == 'training':
            imgs = self.train_imgcodes
        elif self.mode == 'testing':
            imgs = self.test_imgcodes
        return imgs
    
    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index+1) * self.batch_size
        batch_indexes = self.indexes[start_idx:end_idx]
        batch_filenames = [self.filenames[i] for i in batch_indexes]
        x = np.empty((len(batch_indexes), *self.data_shape))
        y = np.empty((len(batch_indexes), *self.data_shape))
        for i, idx in enumerate(batch_indexes):
            image = os.path.join(self.input_data_dir, self.mode, batch_filenames[i] + '.nii.gz')
            img = nib.load(image)
            data = img.get_fdata()
            x[i,:,:,:,0] = data
            y[i,:,:,:,0] = data
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.indexes)


class T1CAEModel(BaseT1CAE):

    '''AI model using ANTsPyNet convolutional autoencoder'''

    def __init__(self, batch_size, epochs):
        super(T1CAEModel, self).__init__()
        self.input_shape = (142, 144, 180, 1)
        self.batch_size = batch_size
        self.epochs = epochs
        # mixed precision for training trades computation time for memory
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

    def build_model(self):
        # use ANTsPyNet 3D CAE
        cae = create_convolutional_autoencoder_model_3d(input_image_size=self.input_shape)
        inputdata = cae.input
        features = cae.output
        print(inputdata.shape)
        print(features.shape)
        model = Model(inputs=inputdata, outputs=dense4, name='T1CAE')
        self.compile_model(model)
        return model

    def compile_model(self, model): 
        mse = MeanSquaredError(reduction='sum_over_batch_size', name='MSE')
        model.compile(loss=mse, optimizer='adam', metrics=['accuracy'])

    def train_model(self, model):
        train_generator = DataGenerator(batch_size=self.batch_size, mode='training')
        cb = EarlyStopping(monitor='loss', verbose=1, patience=3, start_from_epoch=5)
        model.fit(train_generator, epochs=self.epochs, callbacks=cb)

    def test_model(self, model):
        test_generator = DataGenerator(batch_size=self.batch_size, mode='testing')
        loss, accuracy = model.evaluate(test_generator)
        print(f'Testing loss: {loss:.8f}')
        print(f'Testing accuracy: {accuracy:.8f}')


if __name__ == '__main__':
    t1cae_model = T1CAEModel(14, 50)
    model = t1cae_model.build_model()
    #print(model.summary())
    t1cae_model.train_model(model)
    t1cae_model.test_model(model)

