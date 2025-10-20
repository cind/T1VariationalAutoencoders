import os, ast, gc, shutil, random, joblib, logging
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import sklearn.metrics as skm
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import LocallyLinearEmbedding
import umap

logger = logging.getLogger(__name__)


class DatasetNeuropath:
    
    """
    Dataset builder for general predictive models on ADNI data: neuropathology edition.
    Use discovery data cohort only.
    Input (x_data): encodings from UDIP model trained on ADNI dev cohort.
    Target (y_data): measured covariates from original data + neuropathology report.
    """

    def __init__(self, fmap_size):
        self.base_dir = os.getcwd()
        self.label_dir = os.path.join(self.base_dir, 'encodings')
        self.fmap_size = fmap_size
        self.npath_datafile = os.path.join(self.label_dir, 'udips+neuropath.csv')
        self.data = pd.read_csv(self.npath_datafile)
        self.covars = ['Sex', 'APOE4', 'DX', 'Amyloid', 'AdniPhase', 'YrsEdu', 'Age', 'CDRSB', 'Centiloids', 'NPNEUR', 'NPBRAAK', 'NPADNC']
        self.train = self.data.loc[self.data.DiscCohort=='train']
        self.val = self.data.loc[self.data.DiscCohort=='val']
        self.test = self.data.loc[self.data.DiscCohort=='test']
        self.train_imgcodes = list(self.train.T1Code)
        self.val_imgcodes = list(self.val.T1Code)
        self.test_imgcodes = list(self.test.T1Code)
        self.xtr, self.ytr, self.xv, self.yv, self.xte, self.yte = self.get_train_test_data()
    
    def get_train_test_data(self):
        """Main function for getting train/test datasets"""
        # x data
        xtr = self.get_encoding_data('train')
        xv = self.get_encoding_data('val')
        xte = self.get_encoding_data('test')
        # y data
        ytr = self.get_label_data('train')
        yv = self.get_label_data('val')
        yte = self.get_label_data('test')
        return xtr, ytr, xv, yv, xte, yte
    
    def get_label_data(self, mode):
        """Generate y data from covariates"""
        if mode == 'train':
            imgcodes = self.train_imgcodes
            udip_labels = self.train
        elif mode == 'val': 
            imgcodes = self.val_imgcodes
            udip_labels = self.val
        elif mode == 'test':
            imgcodes = self.test_imgcodes
            udip_labels = self.test
        elif mode == 'explore':
            udip_labels = self.data
            imgcodes = list(self.data.T1Code)
        y = np.empty((len(imgcodes), len(self.covars)))
        for i, ix in enumerate(udip_labels.index.values):
            y[i,0] = udip_labels.at[ix,'Sex']
            y[i,1] = udip_labels.at[ix,'APOE4']
            y[i,2] = udip_labels.at[ix,'DX']
            y[i,3] = udip_labels.at[ix,'Amyloid']
            y[i,4] = udip_labels.at[ix,'AdniPhase']
            y[i,5] = udip_labels.at[ix,'YrsEdu']
            y[i,6] = udip_labels.at[ix,'Age']
            y[i,7] = udip_labels.at[ix,'CDRSB']
            y[i,8] = udip_labels.at[ix,'Centiloids']
            y[i,9] = udip_labels.at[ix,'NPNEUR']
            y[i,10] = udip_labels.at[ix,'NPBRAAK']
            y[i,11] = udip_labels.at[ix,'NPADNC']
        return y
    
    def get_encoding_data(self, mode):
        """Generate x data from udip encodings"""
        if mode == 'train':
            imgcodes = self.train_imgcodes
            udip_labels = self.train
        elif mode == 'val':
            imgcodes = self.val_imgcodes
            udip_labels = self.val
        elif mode == 'test':
            imgcodes = self.test_imgcodes
            udip_labels = self.test
        elif mode == 'explore':
            udip_labels = self.data
            imgcodes = list(self.data.T1Code)
        x = np.empty((len(imgcodes), self.fmap_size))
        if self.fmap_size == 256:
            encoding_col = 'UDIP_256'
        elif self.fmap_size == 128:
            encoding_col = 'UDIP'
        for i, ix in enumerate(udip_labels.index.values):
            enc = udip_labels.at[ix,encoding_col]
            x[i,:] = np.array(ast.literal_eval(enc))
        return x    
    

class PredictiveModel:
    """
    ML models using lower-dimensional latent encoding derived from trained autoencoders.
    Input is UDIP encodings on discovery cohort. These encodings were generated from a model trained on ADNI dev cohort.
    """

    def __init__(self, fmap_size):
        self.base_dir = os.getcwd()
        self.label_dir = os.path.join(self.base_dir, 'encodings')
        self.fmap_size = fmap_size
        self.covars = ['Sex', 'APOE4', 'DX', 'Amyloid', 'AdniPhase', 'YrsEdu', 'Age', 'CDRSB', 'Centiloids', 'NPNEUR', 'NPBRAAK', 'NPADNC']
        self.dataset = DatasetNeuropath(self.fmap_size)
        self.udips = self.dataset.get_encoding_data('explore')
        self.labels = self.dataset.get_label_data('explore')
        self.plot_dir = os.path.join(os.getcwd(), 'model_performance', 'neuropath')
    
    def logistic_regression(self, target_idx, solver='saga', max_iter=2500):
        """Run logistic regression on each categorical target (covariate)."""
        y_train = self.dataset.ytr[:,target_idx]
        y_test = self.dataset.yte[:,target_idx]
        lr = LogisticRegression(multi_class='multinomial', solver=solver, max_iter=max_iter)
        lr.fit(self.dataset.xtr, y_train)
        y_pred = lr.predict(self.dataset.xte)
        y_pred_probs = lr.predict_proba(self.dataset.xte)
        acc = skm.accuracy_score(y_test, y_pred)
        logloss = skm.log_loss(y_test, y_pred_probs, labels=lr.classes_)
        print(f"Accuracy (test): {acc:.3f}")
        print(f"Log-loss (test): {logloss:.3f}")
        return y_pred, lr
    
    def local_lin_embed(self, n_neighbors, n_components):
        """Locally linear embedding on latent space vectors"""
        lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, eigen_solver='dense')
        embedding = lle.fit_transform(self.udips)
        return embedding
    
    def plot_logistic_predictions(self, y_true, y_pred, target, fpath):
        """Plot logistic test predictions vs true for categorical targets"""
        class_names = np.unique(y_true)
        cm = skm.confusion_matrix(y_true, y_pred, labels=class_names, normalize='true')
        disp = skm.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(8,6))
        disp.plot(cmap='Blues', ax=ax, values_format='.2f')
        plt.title(f'Logistic performance on {target}')
        plt.tight_layout()
        fpath = os.path.join(self.plot_dir, fpath)
        plt.savefig(fpath)
        plt.close()
    
    def plot_logistic_feature_importance(self, lr, target_idx, fpath, top_n=10):
        """Plot top important features from logistic regression"""
        coefs = lr.coef_
        n_classes = coefs.shape[0]
        class_names = [f'Class {i}' for i in range(n_classes)]
        feature_names = [f'UDIP {i}' for i in range(128)]
        fig, axes = plt.subplots(n_classes, 1, figsize=(10,4*n_classes))
        if not isinstance(axes, (np.ndarray, list)):
            axes = [axes]
        for i, ax in enumerate(axes):
            coef = coefs[i]
            abs_coef = abs(coef)
            top_idx = np.argsort(abs_coef)[-top_n:]
            ax.barh(np.array(feature_names)[top_idx], coef[top_idx], color='skyblue')
            ax.set_title(f'Top {top_n} features for {class_names[i]}', fontsize=12)
            ax.axvline(0, color='gray', linewidth=0.8)
            ax.set_xlabel('Coefficient value')
            ax.tick_params(labelsize=10)
        plt.suptitle(f'Feature importance for logistic regression for target {target_idx}', fontsize=14)
        plt.tight_layout()
        fpath = os.path.join(self.plot_dir, fpath)
        plt.savefig(fpath)
        plt.close()
    
    def run_umap(self, n_neighbors, min_dist, n_components, metric):
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric)
        embedding = reducer.fit_transform(self.udips)
        return embedding
    
    def plot_latent_space(self, x_reduced, fpath, fig_title, labels):
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
        fpath = os.path.join(self.plot_dir, fpath)
        plt.savefig(fpath, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    def pearson_correlation(self, columns):
        """Computes Pearson correlation coefficients and p-values on selected columns"""
        data = self.dataset.data[columns].to_numpy()
        n_vars = data.shape[1]
        corr_matrix = np.zeros((n_vars, n_vars))
        pval_matrix = np.zeros((n_vars, n_vars))
        for i in range(n_vars):
            for j in range(n_vars):
                corr, pval = pearsonr(data[:,i], data[:,j])
                corr_matrix[i,j] = corr
                pval_matrix[i,j] = pval
        corr_df = pd.DataFrame(corr_matrix, index=columns, columns=columns)
        pval_df = pd.DataFrame(pval_matrix, index=columns, columns=columns)
        return corr_df, pval_df

    def plot_corr(corr_df, pval_df=None, sig_lvl=0.05):
        """Plots correlation matrix with optional p-value masking"""
        matrix = corr_df.to_numpy()
        labels = corr_df.columns
        n = len(labels)
        if pval_df is not None:
            mask = pval_df.to_numpy() > sig_lvl
        else:
            mask = np.full_like(matrix, False, dtype=bool)
        fig, ax = plt.subplots(figsize=(1.5*n,1.2*n)) 
        cax = ax.imshow(matrix, cmap='coolwarm', vmin=-1, vmax=1)
        for i in range(n):
            for j in range(n):
                value = matrix[i,j]
                if not mask[i,j]:
                    ax.text(j,i,f'{value:.2f}',va='center',ha='center',color='black')
                else:
                    ax.text(j,i,'x',va='center',ha='center',color='gray',alpha=0.5)
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels,rotation=45,ha='right')
        ax.set_yticklabels(labels)
        ax.set_title('Pearson Correlation Matrix')
        fig.colorbar(cax, ax=ax, shrink=0.75, label='r')
        plt.tight_layout()
        fpath = os.path.join(self.plot_dir, 'corr_udips_neuropath.png')
        plt.savefig(fpath)
    
    def plot_classification_predictions_vs_true(self, y_pred, y_pred_probs, plot_filepath):
        # graph ROC for each ROI after testing
        dim = len(y_pred.shape)
        if dim > 1:
            output_names = self.rois
            num_outputs = len(output_names)
            fig, axes = plt.subplots(nrows=1, ncols=num_outputs, figsize=(15,5))
            plt.subplots_adjust(wspace=0.4)
            for i, output_name in enumerate(output_names):
                fpr, tpr, _ = skm.roc_curve(self.yte[:,i], y_pred_probs[:,i])
                auroc = skm.auc(fpr, tpr)
                axes[i].plot(fpr, tpr, label=f'AUROC={auroc:.2f}')
                axes[i].plot([0,1], [0,1], 'k--', label='Chance level')
                axes[i].set_title('ROC for ' + output_name)
                axes[i].set_xlabel('False positive rate')
                axes[i].set_ylabel('True positive rate')
                axes[i].legend(loc='lower right')
            plt.savefig(plot_filepath)
            plt.clf()
        else:
            roc = skm.RocCurveDisplay.from_predictions(self.yte, y_pred)
            roc.ax_.set_title('AUROC for classifier')
            roc.plot(plot_chance_level=True)
            plt.savefig(plot_filepath)
            plt.clf()
    

if __name__=='__main__':
    gc.collect()
    latent_size = 128
    n = 10
    model = PredictiveModel(latent_size)
    cols = ['UDIP','CDRSB','Centiloids','NPNEUR','NPBRAAK','NPADNC']
    corr_df, pval_df = model.pearson_correlation(cols)
    model.plot_corr(corr_df, pval_df)

    """
    # run LLE and plot
    x_lin = model.local_lin_embed(n_neighbors=n, n_components=2)
    model.plot_latent_space(x_lin, 'lle_apoe.png', 'LLE and APOE4 (# alleles)', labels=model.labels[:,1])
    model.plot_latent_space(x_lin, 'lle_cdr.png', 'LE and CDRSB', labels=model.labels[:,7])
    model.plot_latent_space(x_lin, 'lle_age.png', 'LLE and Age', labels=model.labels[:,6])
    model.plot_latent_space(x_lin, 'lle_sex.png', 'LLE and Sex (1=M, 2=F)', labels=model.labels[:,0])
    model.plot_latent_space(x_lin, 'lle_amyloid.png', 'LLE and Amyloid (0=neg, 1=pos)', labels=model.labels[:,3])
    model.plot_latent_space(x_lin, 'lle_dx.png', 'LLE and Diagnosis (1=CN, 2=MCI, 3=AD)', labels=model.labels[:,2])
    model.plot_latent_space(x_lin, 'lle_edu.png', 'LLE and Education', labels=model.labels[:,5])
    model.plot_latent_space(x_lin, 'lle_adniphase.png', 'LLE and ADNI phase', labels=model.labels[:,4])
    model.plot_latent_space(x_lin, 'lle_centiloids.png', 'LLE and Centiloid', labels=model.labels[:,8])
    model.plot_latent_space(x_lin, 'lle_npneur.png', 'LLE and Neuritic Plaques', labels=model.labels[:,9])
    model.plot_latent_space(x_lin, 'lle_npbraak.png', 'LLE and Tau Braak stage', labels=model.labels[:,10])
    model.plot_latent_space(x_lin, 'lle_npadnc.png', 'LLE and ADNC', labels=model.labels[:,11])

    # run UMAP feature analysis and plot
    x_reduced = model.run_umap(n_neighbors=n, min_dist=0.1, n_components=2, metric='euclidean')
    model.plot_latent_space(x_reduced, 'umap_apoe.png', 'UMAP and APOE4 (# alleles)', labels=model.labels[:,1])
    model.plot_latent_space(x_reduced, 'umap_cdr.png', 'UMAP and CDRSB', labels=model.labels[:,7])
    model.plot_latent_space(x_reduced, 'umap_age.png', 'UMAP and Age', labels=model.labels[:,6])
    model.plot_latent_space(x_reduced, 'umap_sex.png', 'UMAP and Sex (1=M, 2=F)', labels=model.labels[:,0])
    model.plot_latent_space(x_reduced, 'umap_amyloid.png', 'UMAP and Amyloid (0=neg, 1=pos)', labels=model.labels[:,3])
    model.plot_latent_space(x_reduced, 'umap_dx.png', 'UMAP and Diagnosis (1=CN, 2=MCI, 3=AD)', labels=model.labels[:,2])
    model.plot_latent_space(x_reduced, 'umap_edu.png', 'UMAP and Education', labels=model.labels[:,5])
    model.plot_latent_space(x_reduced, 'umap_adniphase.png', 'UMAP and ADNI phase', labels=model.labels[:,4])
    model.plot_latent_space(x_reduced, 'umap_centiloids.png', 'UMAP and Centiloid', labels=model.labels[:,8])
    model.plot_latent_space(x_reduced, 'umap_npneur.png', 'UMAP and Neuritic Plaques', labels=model.labels[:,9])
    model.plot_latent_space(x_reduced, 'umap_npbraak.png', 'UMAP and Tau Braak stage', labels=model.labels[:,10])
    model.plot_latent_space(x_reduced, 'umap_npadnc.png', 'UMAP and ADNC', labels=model.labels[:,11])
"""

