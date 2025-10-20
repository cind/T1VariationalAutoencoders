import os, ast, gc, shutil, random, joblib, logging
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import sklearn.metrics as skm
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.manifold import LocallyLinearEmbedding
import umap

logger = logging.getLogger(__name__)


class DatasetTau:
    
    """
    Dataset builder for general predictive models on ADNI data: tau edition.
    Use discovery data cohort only.
    Input (x_data): encodings from UDIP model trained on ADNI dev cohort.
    Target (y_data): measured covariates from original data + tau PET SUVR data at last time point measured for each subject.
    """

    def __init__(self, fmap_size):
        self.base_dir = os.getcwd()
        self.label_dir = os.path.join(self.base_dir, 'encodings')
        self.fmap_size = fmap_size
        #self.tau_datafile = os.path.join(self.label_dir, 'udips+tau_ltp.csv')
        self.tau_datafile = os.path.join(self.label_dir, 'udips+tau.csv')
        self.data = pd.read_csv(self.tau_datafile)
        self.covars = ['Sex', 'APOE4', 'DX', 'Amyloid', 'AdniPhase', 'YrsEdu', 'Age', 'CDRSB', 'Centiloids', 'MedTempTau', 'MetTempTau', 'TempParTau', 'BinaryMedTempTau', 'BinaryMetTempTau', 'BinaryTempParTau']
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
            y[i,9] = udip_labels.at[ix,'MedTempTau']
            y[i,10] = udip_labels.at[ix,'MetTempTau']
            y[i,11] = udip_labels.at[ix,'TempParTau']
            y[i,12] = udip_labels.at[ix,'BinaryMedTempTau']
            y[i,13] = udip_labels.at[ix,'BinaryMetTempTau']
            y[i,14] = udip_labels.at[ix,'BinaryTempParTau']
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
        self.covars = ['Sex', 'APOE4', 'DX', 'Amyloid', 'AdniPhase', 'YrsEdu', 'Age', 'CDRSB', 'Centiloids', 'MedTempTau', 'MetTempTau', 'TempParTau', 'BinaryMedTempTau', 'BinaryMetTempTau', 'BinaryTempParTau']
        self.dataset = DatasetTau(self.fmap_size)
        self.udips = self.dataset.get_encoding_data('explore')
        self.labels = self.dataset.get_label_data('explore')
        self.plot_dir = os.path.join(os.getcwd(), 'model_performance')
    
    def logistic_regression(self, target_idx, udip_idxs=None, solver='saga', max_iter=2500):
        """Run logistic regression on binary tau targets."""
        if udip_idxs is not None:
            x_train = self.dataset.xtr[:,udip_idxs]
            x_test = self.dataset.xte[:,udip_idxs]
        else:
            x_train = self.dataset.xtr
            x_test = self.dataset.xte
        y_train = self.dataset.ytr[:,target_idx]
        y_test = self.dataset.yte[:,target_idx]
        lr = LogisticRegression(solver=solver, max_iter=max_iter)
        lr.fit(x_train, y_train)
        y_pred = lr.predict(x_test)
        y_pred_probs = lr.predict_proba(x_test)
        acc = skm.accuracy_score(y_test, y_pred)
        f1 = skm.f1_score(y_test, y_pred)
        logloss = skm.log_loss(y_test, y_pred_probs, labels=lr.classes_)
        print(f"Accuracy (test): {acc:.3f}")
        print(f"F1 score (test): {f1:.3f}")
        print(f"Log-loss (test): {logloss:.3f}")
        return y_test, y_pred, lr
    
    def local_lin_embed(self, n_neighbors, n_components):
        """Locally linear embedding on latent space vectors"""
        lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, eigen_solver='dense')
        embedding = lle.fit_transform(self.udips)
        return embedding
    
    def plot_logistic_predictions(self, y_true, y_pred, fpath):
        """Plot logistic test predictions vs true for binary tau targets"""
        class_names = np.unique(y_true)
        cm = skm.confusion_matrix(y_true, y_pred, labels=class_names, normalize='true')
        disp = skm.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(8,6))
        disp.plot(cmap='Blues', ax=ax, values_format='.2f')
        plt.title(f'Logistic model performance')
        plt.tight_layout()
        fpath = os.path.join(self.plot_dir, fpath)
        plt.savefig(fpath)
        plt.close()
    
    def plot_logistic_feature_importance(self, lr, target_idx, fpath, top_n=10):
        """Plot top important features from logistic regression"""
        coefs = lr.coef_
        n_classes = coefs.shape[0]
        class_names = [f'Class {i}' for i in range(n_classes)]
        feature_names = [f'UDIP {i}' for i in range(self.fmap_size)]
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
    
    def linear_regression(self, target_idx, udip_idxs=None):
        """Run ordinary least squares (linear) regression on continuous tau targets."""
        if udip_idxs is not None:
            x_train = self.dataset.xtr[:,udip_idxs]
            x_test = self.dataset.xte[:,udip_idxs]
        else:
            x_train = self.dataset.xtr
            x_test = self.dataset.xte
        y_train = self.dataset.ytr[:,target_idx]
        y_test = self.dataset.yte[:,target_idx]
        lin = LinearRegression()
        lin.fit(x_train, y_train)
        y_pred = lin.predict(x_test)
        r2 = skm.r2_score(y_test, y_pred)
        rmse = np.sqrt(skm.mean_squared_error(y_test, y_pred))
        print(f"R² score (test): {r2:.3f}")
        print(f"RMSE (test): {rmse:.3f}")
        return y_test, y_pred, lin

    def pls_regression(self, target_idx, udip_idxs=None, n_comp=4):    
        """Run PLS regression on continuous tau targets."""
        if udip_idxs is not None:
            x_train = self.dataset.xtr[:,udip_idxs]
            x_test = self.dataset.xte[:,udip_idxs]
        else:
            x_train = self.dataset.xtr
            x_test = self.dataset.xte
        y_train = self.dataset.ytr[:,target_idx]
        y_test = self.dataset.yte[:,target_idx]
        pls = PLSRegression(n_components=n_comp)
        pls.fit(x_train, y_train)
        y_pred = pls.predict(x_test)
        r2 = skm.r2_score(y_test, y_pred)
        rmse = np.sqrt(skm.mean_squared_error(y_test, y_pred))
        print(f"R² score (test): {r2:.3f}")
        print(f"RMSE (test): {rmse:.3f}")
        return y_test, y_pred, pls

    def elasticnet_regression(self, target_idx, udip_idxs=None):
        """Run ElasticNet regression on continuous tau targets."""
        if udip_idxs is not None:
            x_train = self.dataset.xtr[:,udip_idxs]
            x_test = self.dataset.xte[:,udip_idxs]
        else:
            x_train = self.dataset.xtr
            x_test = self.dataset.xte
        y_train = self.dataset.ytr[:,target_idx]
        y_test = self.dataset.yte[:,target_idx]
        eln = ElasticNet()
        eln.fit(x_train, y_train)
        y_pred = eln.predict(x_test)
        r2 = skm.r2_score(y_test, y_pred)
        rmse = np.sqrt(skm.mean_squared_error(y_test, y_pred))
        print(f"R² score (test): {r2:.3f}")
        print(f"RMSE (test): {rmse:.3f}")
        return y_test, y_pred, eln

    def plot_regression_predictions(self, y_true, y_pred, target, fpath, model_type):
        """Plot regression test predictions vs true for continuous tau targets."""
        Y_true = np.asarray(y_true)
        Y_pred = np.asarray(y_pred)
        Y_true = Y_true.reshape(-1, 1)
        Y_pred = Y_pred.reshape(-1, 1)
        fig, ax = plt.subplots(1, 1, figsize=(5,4), squeeze=False)
        ax[0,0].scatter(Y_true, Y_pred, alpha=0.7, edgecolors='k')
        ax[0,0].plot([Y_true.min(), Y_true.max()], [Y_true.min(), Y_true.max()], 'r--', lw=2)
        ax[0,0].set_xlabel(f"True {target}")
        ax[0,0].set_ylabel(f"Predicted {target}")
        ax[0,0].set_title(f"{model_type} performance on {target}")
        ax[0,0].grid(True)
        plt.tight_layout()
        fpath = os.path.join(self.plot_dir, fpath)
        plt.savefig(fpath)
        plt.close()

    def run_umap(self, n_neighbors, min_dist, n_components, metric):
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric)
        embedding = reducer.fit_transform(self.udips)
        return embedding
    
    def plot_latent_space(self, x_reduced, fpath, fig_title, labels):
        """Plots dimensionality reduction to 2D/3D of latent space vectors colored by labels"""
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
    
    def pearson_correlation(self):
        """Computes Pearson correlation coefficients and p-values on selected columns"""
        x_cols = ['UDIP'+str(i) for i in range(self.fmap_size)]
        y_cols = ['MedTemp', 'MetTemp', 'TempPar']
        x = self.udips
        y = self.labels[:,9:12]
        corrs = np.zeros((len(x_cols), len(y_cols)))
        pvals = np.zeros((len(x_cols), len(y_cols)))
        for i, x_col in enumerate(x_cols):
            for j, y_col in enumerate(y_cols):
                corr, pval = pearsonr(x[:,i], y[:,j])
                corrs[i,j] = corr
                pvals[i,j] = pval
        return corrs, pvals

    def get_sig_high_corr(self, corrs, pvals, corr_thr=0.15, pval_thr=0.05):
        """Filter correlation matrix for significant p-values
           and rows with at least one value > threshold.""" 
        mask_pval = pvals < pval_thr
        mask_corr = abs(corrs) > corr_thr
        mask = mask_corr & mask_pval
        rows = mask.any(axis=1)
        corrs = corrs[rows,:]
        pvals = pvals[rows,:]
        idxs = np.nonzero(rows)[0]
        return corrs, pvals, idxs

    def plot_corr(self, corrs, pvals, idxs):
        """Plots correlation matrix with filtering"""
        n_rows, n_cols = corrs.shape
        x_labels = ['UDIP'+str(i) for i in idxs]
        y_labels = ['MedTemp', 'MetTemp', 'TempPar']
        fig, ax = plt.subplots(figsize=(2*n_cols, n_rows)) 
        cax = ax.imshow(corrs, cmap='coolwarm', vmin=-1, vmax=1)
        for i in range(n_rows):
            for j in range(n_cols):
                val = corrs[i,j]
                ax.text(j,i,f'{val:.2f}',va='center',ha='center',color='black',fontsize=12)
        ax.set_xticks(np.arange(n_cols))
        ax.set_yticks(np.arange(n_rows))
        ax.set_xticklabels(y_labels,rotation=45,ha='right',fontsize=10)
        ax.set_yticklabels(x_labels,fontsize=10)
        ax.set_title('Pearson Correlation Matrix')
        fig.colorbar(cax, ax=ax, shrink=0.75, label='r')
        plt.tight_layout()
        fpath = os.path.join(self.plot_dir, 'corr_udips'+str(self.fmap_size)+'_tau.png')
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
    latent_size = 256
    n = 50
    corr_thr = 0.2
    model = PredictiveModel(latent_size)
    tau_rois = ['MedTemp','MetTemp','TempPar']
    binary_tau_idxs = [12,13,14]
    cont_tau_idxs = [9,10,11]

    # get correlation between udips and continuous tau labels
    corrs, pvals = model.pearson_correlation()
    corrs, pvals, idxs = model.get_sig_high_corr(corrs, pvals, corr_thr)
    model.plot_corr(corrs, pvals, idxs)

'''
    # run logistic regression between udips and binary tau labels
    for ix, i in enumerate(binary_tau_idxs):
        y_true, y_pred, lr = model.logistic_regression(i)
        model.plot_logistic_predictions(y_true, y_pred, 'lr_tau'+tau_rois[ix]+'_udips.png')

    # run regression between udips and continuous tau labels
    for ix, i in enumerate(cont_tau_idxs):
        # ordinary least squares linear regression
        y_true, y_pred, lin = model.linear_regression(i)
        model.plot_regression_predictions(y_true, y_pred, tau_rois[ix], 'lin_tau'+tau_rois[ix]+'_udips.png', 'Linear')
        # partial least squares linear regression
        y_true, y_pred, pls = model.pls_regression(i)
        model.plot_regression_predictions(y_true, y_pred, tau_rois[ix], 'pls_tau'+tau_rois[ix]+'_udips.png', 'PLS')
        # elasticnet regression
        y_true, y_pred, eln = model.elasticnet_regression(i)
        model.plot_regression_predictions(y_true, y_pred, tau_rois[ix], 'eln_tau'+tau_rois[ix]+'_udips.png', 'ElasticNet')     
    
    # run LLE and plot
    x_lin = model.local_lin_embed(n_neighbors=n, n_components=3)
    model.plot_latent_space(x_lin, 'lle_apoe.png', 'LLE and APOE4 (# alleles)', labels=model.labels[:,1])
    model.plot_latent_space(x_lin, 'lle_cdr.png', 'LLE and CDRSB', labels=model.labels[:,7])
    model.plot_latent_space(x_lin, 'lle_age.png', 'LLE and Age', labels=model.labels[:,6])
    model.plot_latent_space(x_lin, 'lle_sex.png', 'LLE and Sex (1=M, 2=F)', labels=model.labels[:,0])
    model.plot_latent_space(x_lin, 'lle_amyloid.png', 'LLE and Amyloid (0=neg, 1=pos)', labels=model.labels[:,3])
    model.plot_latent_space(x_lin, 'lle_dx.png', 'LLE and Diagnosis (1=CN, 2=MCI, 3=AD)', labels=model.labels[:,2])
    model.plot_latent_space(x_lin, 'lle_edu.png', 'LLE and Education', labels=model.labels[:,5])
    model.plot_latent_space(x_lin, 'lle_adniphase.png', 'LLE and ADNI phase', labels=model.labels[:,4])
    model.plot_latent_space(x_lin, 'lle_centiloids.png', 'LLE and Centiloid', labels=model.labels[:,8])
    model.plot_latent_space(x_lin, 'lle_medtemptau.png', 'LLE and MedTempTau', labels=model.labels[:,9])
    model.plot_latent_space(x_lin, 'lle_mettemptau.png', 'LLE and MetTempTau', labels=model.labels[:,10])
    model.plot_latent_space(x_lin, 'lle_temppartau.png', 'LLE and TempParTau', labels=model.labels[:,11])
    model.plot_latent_space(x_lin, 'lle_binmedtemptau.png', 'LLE and BinaryMedTempTau', labels=model.labels[:,12])
    model.plot_latent_space(x_lin, 'lle_binmettemptau.png', 'LLE and BinaryMetTempTau', labels=model.labels[:,13])
    model.plot_latent_space(x_lin, 'lle_bintemppartau.png', 'LLE and BinaryTempParTau', labels=model.labels[:,14])
    
    # run UMAP feature analysis and plot
    x_reduced = model.run_umap(n_neighbors=n, min_dist=0.2, n_components=3, metric='euclidean')
    model.plot_latent_space(x_reduced, 'umap_apoe.png', 'UMAP and APOE4 (# alleles)', labels=model.labels[:,1])
    model.plot_latent_space(x_reduced, 'umap_cdr.png', 'UMAP and CDRSB', labels=model.labels[:,7])
    model.plot_latent_space(x_reduced, 'umap_age.png', 'UMAP and Age', labels=model.labels[:,6])
    model.plot_latent_space(x_reduced, 'umap_sex.png', 'UMAP and Sex (1=M, 2=F)', labels=model.labels[:,0])
    model.plot_latent_space(x_reduced, 'umap_amyloid.png', 'UMAP and Amyloid (0=neg, 1=pos)', labels=model.labels[:,3])
    model.plot_latent_space(x_reduced, 'umap_dx.png', 'UMAP and Diagnosis (1=CN, 2=MCI, 3=AD)', labels=model.labels[:,2])
    model.plot_latent_space(x_reduced, 'umap_edu.png', 'UMAP and Education', labels=model.labels[:,5])
    model.plot_latent_space(x_reduced, 'umap_adniphase.png', 'UMAP and ADNI phase', labels=model.labels[:,4])
    model.plot_latent_space(x_reduced, 'umap_centiloids.png', 'UMAP and Centiloid', labels=model.labels[:,8])
    model.plot_latent_space(x_reduced, 'umap_medtemptau.png', 'UMAP and MedTempTau', labels=model.labels[:,9])
    model.plot_latent_space(x_reduced, 'umap_mettemptau.png', 'UMAP and MetTempTau', labels=model.labels[:,10])
    model.plot_latent_space(x_reduced, 'umap_temppartau.png', 'UMAP and TempParTau', labels=model.labels[:,11])
    model.plot_latent_space(x_reduced, 'umap_binmedtemptau.png', 'UMAP and BinaryMedTempTau', labels=model.labels[:,12])
    model.plot_latent_space(x_reduced, 'umap_binmettemptau.png', 'UMAP and BinaryMetTempTau', labels=model.labels[:,13])
    model.plot_latent_space(x_reduced, 'umap_bintemppartau.png', 'UMAP and BinaryTempParTau', labels=model.labels[:,14])
'''

