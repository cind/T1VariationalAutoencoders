import os, ast, gc, shutil, random, joblib, logging
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import sklearn.metrics as skm

logger = logging.getLogger(__name__)


class DatasetADNI:
    """Build dataset for UMAP analysis"""

    def __init__(self, encoding_col):
        self.base_dir = os.getcwd()
        self.encoding_col = encoding_col
        self.encodings_file = os.path.join(self.base_dir, 'encodings', 'adni_encodings.csv')
        self.encodings = pd.read_csv(self.encodings_file)
        self.covars = ['APOE4', 'CDRSB', 'Age', 'Sex', 'Amyloid', 'DX']
        self.fmap_size = 128
    
    def get_encoding_data(self):
        """For encoding only predictions + analysis"""
        imgcodes = list(self.encodings.T1Code)
        x = np.empty((len(imgcodes), self.fmap_size))
        for i, ix in enumerate(self.encodings.index.values):
            enc = self.encodings.at[ix,self.encoding_col]
            x[i,:] = np.array(ast.literal_eval(enc))
        return x    
    
    def get_combined_data(self):
        """For combined encoding + covars"""
        imgcodes = list(self.encodings.T1Code)
        sz = self.fmap_size + len(self.covars)
        x = np.empty((len(imgcodes), sz))
        for i, ix in enumerate(self.encodings.index.values):
            enc = self.encodings.at[ix,self.encoding_col]
            x[i,0:128] = np.array(ast.literal_eval(enc))
            x[i,128] = self.encodings.at[ix,'APOE4']
            x[i,129] = self.encodings.at[ix,'CDRSB']
            x[i,130] = self.encodings.at[ix,'Age']
            x[i,131] = self.encodings.at[ix,'Sex']
            x[i,132] = self.encodings.at[ix,'Amyloid']
            x[i,133] = self.encodings.at[ix,'DX']
        return x
        

class UmapADNI:
    """Run UMAP analysis"""

    def __init__(self):
        self.base_dir = os.getcwd()
        self.plot_dir = os.path.join(self.base_dir, 'model_performance', 'umaps')
        self.ds_udip_pretrained = DatasetADNI('UDIP')
        self.ds_udip_trainADNI = DatasetADNI('UDIP_ADNI')
        self.ds_ants_trainADNI = DatasetADNI('VectorEncoding')
        self.encodings = self.ds_udip_pretrained.encodings
        self.x_udip_pretrained = self.ds_udip_pretrained.get_combined_data()
        self.x_udip_trainADNI = self.ds_udip_trainADNI.get_combined_data()
        self.x_ants_trainADNI = self.ds_ants_trainADNI.get_combined_data()
    
    def run_umap(self, x, n_neighbors, min_dist, n_components, metric):
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric)
        x_enc = x[:,:128]
        embedding = reducer.fit_transform(x_enc)
        return embedding
    
    def plot_latent_space(self, x_reduced, filepath, fig_title, labels):
        """Plots dimensionality reduction to 2D of latent space vectors"""
        fig = plt.figure()
        if len(x_reduced[0,:]) == 2:
            ax = fig.add_subplot(111)
            s = ax.scatter(x_reduced[:,0], x_reduced[:,1], c=labels, s=20)
        elif len(x_reduced[0,:]) == 3:
            ax = fig.add_subplot(111, projection='3d')
            s = ax.scatter(x_reduced[:,0], x_reduced[:,1], x_reduced[:,2], c=labels, s=50)
        plt.colorbar(s) 
        fig.suptitle(fig_title)
        fig.tight_layout()
        #plt.show()
        plt.savefig(os.path.join(self.plot_dir, filepath), bbox_inches='tight', pad_inches=0)
    

if __name__=='__main__':
    gc.collect()
    
    # UMAP feature analysis
    m = UmapADNI()
    x_proj_udip_pretrained = m.run_umap(m.x_udip_pretrained, n_neighbors=100, min_dist=0.2, n_components=3, metric='euclidean')
    x_proj_udip_trainADNI = m.run_umap(m.x_udip_trainADNI, n_neighbors=100, min_dist=0.2, n_components=3, metric='euclidean')
    x_proj_ants_trainADNI = m.run_umap(m.x_ants_trainADNI, n_neighbors=100, min_dist=0.2, n_components=3, metric='euclidean')
    
    # plot each UMAP colored by covariates - pretrained UDIP model
    m.plot_latent_space(x_proj_udip_pretrained, 'udip_umap_dx.png', 'UDIP (pretrained) UMAP and Diagnosis (1=CN, 2=MCI, 3=AD)', labels=m.encodings.DX)
    m.plot_latent_space(x_proj_udip_pretrained, 'udip_umap_cdr.png', 'UDIP (pretrained) UMAP and CDRSB', labels=m.encodings.CDRSB)
    m.plot_latent_space(x_proj_udip_pretrained, 'udip_umap_apoe.png', 'UDIP (pretrained) UMAP and APOE4 (# alleles)', labels=m.encodings.APOE4)
    m.plot_latent_space(x_proj_udip_pretrained, 'udip_umap_age.png', 'UDIP (pretrained) UMAP and Age', labels=m.encodings.Age)
    m.plot_latent_space(x_proj_udip_pretrained, 'udip_umap_sex.png', 'UDIP (pretrained) UMAP and Sex (1=M, 2=F)', labels=m.encodings.Sex)
    m.plot_latent_space(x_proj_udip_pretrained, 'udip_umap_amyloid.png', 'UDIP (pretrained) UMAP and Amyloid (0=neg, 1=pos)', labels=m.encodings.Amyloid)

    # plot each UMAP colored by covariates - UDIP model trained on ADNI data
    m.plot_latent_space(x_proj_udip_trainADNI, 'udipadni_umap_dx.png', 'UDIP (ADNI trained) UMAP and Diagnosis (1=CN, 2=MCI, 3=AD)', labels=m.encodings.DX)
    m.plot_latent_space(x_proj_udip_trainADNI, 'udipadni_umap_cdr.png', 'UDIP (ADNI trained) UMAP and CDRSB', labels=m.encodings.CDRSB)
    m.plot_latent_space(x_proj_udip_trainADNI, 'udipadni_umap_apoe.png', 'UDIP (ADNI trained) UMAP and APOE4 (# alleles)', labels=m.encodings.APOE4)
    m.plot_latent_space(x_proj_udip_trainADNI, 'udipadni_umap_age.png', 'UDIP (ADNI trained) UMAP and Age', labels=m.encodings.Age)
    m.plot_latent_space(x_proj_udip_trainADNI, 'udipadni_umap_sex.png', 'UDIP (ADNI trained) UMAP and Sex (1=M, 2=F)', labels=m.encodings.Sex)
    m.plot_latent_space(x_proj_udip_trainADNI, 'udipadni_umap_amyloid.png', 'UDIP (ADNI trained) UMAP and Amyloid (0=neg, 1=pos)', labels=m.encodings.Amyloid)

    # plot each UMAP colored by covariates - ants CAE trained on ADNI data
    m.plot_latent_space(x_proj_ants_trainADNI, 'ants_umap_dx.png', 'ANTs UMAP and Diagnosis (1=CN, 2=MCI, 3=AD)', labels=m.encodings.DX)
    m.plot_latent_space(x_proj_ants_trainADNI, 'ants_umap_cdr.png', 'ANTs UMAP and CDRSB', labels=m.encodings.CDRSB)
    m.plot_latent_space(x_proj_ants_trainADNI, 'ants_umap_apoe.png', 'ANTs UMAP and APOE4 (# alleles)', labels=m.encodings.APOE4)
    m.plot_latent_space(x_proj_ants_trainADNI, 'ants_umap_age.png', 'ANTs UMAP and Age', labels=m.encodings.Age)
    m.plot_latent_space(x_proj_ants_trainADNI, 'ants_umap_sex.png', 'ANTs UMAP and Sex (1=M, 2=F)', labels=m.encodings.Sex)
    m.plot_latent_space(x_proj_ants_trainADNI, 'ants_umap_amyloid.png', 'ANTs UMAP and Amyloid (0=neg, 1=pos)', labels=m.encodings.Amyloid)

