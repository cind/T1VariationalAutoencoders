import os, ast, gc, shutil, random, joblib, logging
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import sklearn.metrics as skm

logger = logging.getLogger(__name__)

base_dir = os.getcwd()
encoded_data_file = os.path.join(base_dir, 'fmap10_T1_vector_encoding.csv')
encoded_data = pd.read_csv(encoded_data_file)
imgcodes = list(encoded_data['ImgCode'])

def get_data(fmap_size):
    x = np.empty((len(imgcodes), fmap_size))
    y = np.empty((len(imgcodes), 2))
    for i in encoded_data.index:
        enc = encoded_data.at[i,'VectorEncoding']
        x[i,:] = np.array(ast.literal_eval(enc))
        dx = encoded_data.at[i,'DX']
        if dx == 'NoDX':
            y[i,0] = 0
        elif dx == 'CN':
            y[i,0] = 0.33
        elif dx == 'MCI':
            y[i,0] = 0.66
        elif dx == 'AD': 
            y[i,0] = 1
        amy = encoded_data.at[i,'Amyloid']
        if amy == 'AB-' or amy == 'Unknown':
            y[i,1] = 0
        elif amy == 'ABConverter':
            y[i,1] = 0.5
        elif amy == 'AB+':
            y[i,1] = 1
    return x, y
        
def split_train_test(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=666)
    return x_train, x_test, y_train, y_test

def pca(x):
    x = StandardScaler().fit_transform(x)
    pc = PCA(n_components=2)
    x_pc = pc.fit_transform(x)
    #print('Explained variance ratio:', pc.explained_variance_ratio_)
    return x_pc
    
def tsne(x):
    x = StandardScaler().fit_transform(x)
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
    x_2dim = tsne.fit_transform(x)
    return x_2dim

def plot_latent_space(x_reduced, filepath, fig_title, labels=None):
    plt.figure(figsize=(8,6))
    if labels is not None:
        s = plt.scatter(x_reduced[:,0], x_reduced[:,1], c=labels)
        plt.colorbar(s)
    else:
        plt.scatter(x_reduced[:,0], x_reduced[:,1])
    plt.title(fig_title)
    plt.xlabel('dim2')
    plt.ylabel('dim2')
    plt.savefig(filepath)
    
def kmeans(data, k):
    clusters = KMeans(n_clusters=k, init='random', n_init=10, verbose=1, random_state=42).fit_predict(data)
    return clusters

def plot_kmeans(data, clusters, filepath):
    plt.scatter(data[:,0], data[:,1], c=clusters)
    plt.title('K-Means clusters')
    plt.savefig(filepath)
    
gc.collect()
wd = os.getcwd()
x, y = get_data(10)
x_pca = pca(x)
x_tsne = tsne(x)
clusters = kmeans(x, 3)
plot_latent_space(x_pca, 'pca.png', 'PCA of latent vectors', labels=y[:,0])
plot_latent_space(x_tsne, 'tsne.png', 'TSNE of latent vectors', labels=y[:,0])
plot_kmeans(x, clusters, 'kmeans3.png')

