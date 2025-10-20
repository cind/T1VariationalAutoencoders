import os, ast, gc, shutil, random, joblib, logging
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import sklearn.metrics as skm
from scipy.stats import pearsonr
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.manifold import LocallyLinearEmbedding
import umap

logger = logging.getLogger(__name__)


class DatasetADNI:
    
    """
    Dataset builder for general predictive models on ADNI data.
    Use discovery data cohort only.
    Input (x_data): embeddings trained on ADNI dev cohort.
    Target (y_data): measured covariates.
    """

    def __init__(self, fmap_size):
        self.base_dir = os.getcwd()
        self.label_dir = os.path.join(self.base_dir, 'encodings')
        self.fmap_size = fmap_size
        #self.datafile = os.path.join(self.label_dir, 'udip_data_covars.csv')
        #self.datafile = os.path.join(self.label_dir, 'adni_udip_covars_wmh.csv')
        self.datafile = os.path.join(self.label_dir, 'discdata_covars_latents.csv')
        self.data = pd.read_csv(self.datafile)
        #self.covars = ['Sex', 'APOE4', 'DX', 'Amyloid', 'AdniPhase', 'YrsEdu', 'Age', 'CDRSB', 'Centiloids']
        self.covars = ['Sex', 'APOE4', 'DX', 'Amyloid', 'Age', 'LogCDRSB', 'Centiloids', 'LogNormWMH', 'NormGM', 'NormWM', 'NormCSF']
        #self.train = self.data.loc[self.data.DiscCohort=='train']
        #self.val = self.data.loc[self.data.DiscCohort=='val']
        #self.test = self.data.loc[self.data.DiscCohort=='test']
        #self.train_imgcodes = list(self.train.T1Code)
        #self.val_imgcodes = list(self.val.T1Code)
        #self.test_imgcodes = list(self.test.T1Code)
        #self.xtr, self.ytr, self.xv, self.yv, self.xte, self.yte = self.get_train_test_data()
    
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
            y[i,4] = udip_labels.at[ix,'Age']
            y[i,5] = udip_labels.at[ix,'LogCDRSB']
            y[i,6] = udip_labels.at[ix,'Centiloids']
            y[i,7] = udip_labels.at[ix,'LogNormWMH']
            y[i,8] = udip_labels.at[ix,'NormGM']
            y[i,9] = udip_labels.at[ix,'NormWM']
            y[i,10] = udip_labels.at[ix,'NormCSF']
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
            encoding_col = 'Embedding'
        for i, ix in enumerate(udip_labels.index.values):
            enc = udip_labels.at[ix,encoding_col]
            x[i,:] = np.array(ast.literal_eval(enc))
        return x    
    

class PredictiveModel:
    """
    ML models using lower-dimensional latent encoding derived from trained autoencoders.
    Input is encodings on discovery cohort. These encodings were generated from a model trained on ADNI dev cohort.
    """

    def __init__(self, fmap_size):
        self.base_dir = os.getcwd()
        self.label_dir = os.path.join(self.base_dir, 'encodings')
        self.fmap_size = fmap_size
        #self.covars = ['Sex', 'APOE4', 'DX', 'Amyloid', 'AdniPhase', 'YrsEdu', 'Age', 'CDRSB', 'Centiloids']
        self.covars = ['Sex', 'APOE4', 'DX', 'Amyloid', 'Age', 'LogCDRSB', 'Centiloids', 'LogNormWMH', 'NormGM', 'NormWM', 'NormCSF']
        #self.covars_continuous = ['YrsEdu', 'Age', 'CDRSB', 'Centiloids']
        #self.covars_categorical = ['Sex', 'APOE4', 'DX', 'Amyloid', 'AdniPhase']
        self.dataset = DatasetADNI(self.fmap_size)
        #self.dataset256 = DatasetADNI(256)
        self.udips = self.dataset.get_encoding_data('explore')
        #self.udips256 = self.dataset256.get_encoding_data('explore')
        self.labels = self.dataset.get_label_data('explore')
        self.plot_dir = os.path.join(os.getcwd(), 'model_performance')
    
    def pearson_correlation(self):
        """Computes Pearson correlation coefficients and p-values between 128-dim and 256-dim encodings"""
        x_cols = ['UDIP'+str(i) for i in range(128)]
        y_cols = ['UDIP'+str(i) for i in range(256)]
        x = self.udips
        y = self.udips256
        corrs = np.zeros((len(x_cols), len(y_cols)))
        pvals = np.zeros((len(x_cols), len(y_cols)))
        for i, x_col in enumerate(x_cols):
            for j, y_col in enumerate(y_cols):
                corr, pval = pearsonr(x[:,i], y[:,j])
                corrs[i,j] = corr
                pvals[i,j] = pval
        return corrs, pvals

    def plot_corr(self, corrs, pvals, sig_lvl=0.01):
        """Plots correlation matrix with p-value masking"""
        n_rows, n_cols = corrs.shape
        x_labels = ['UDIP'+str(i) for i in range(128)]
        y_labels = ['UDIP'+str(i) for i in range(256)]
        mask = pvals > sig_lvl
        fig, ax = plt.subplots(figsize=(0.75*n_cols,0.75*n_rows)) 
        cax = ax.imshow(corrs, cmap='coolwarm', vmin=-1, vmax=1)
        for i in range(n_rows):
            for j in range(n_cols):
                val = corrs[i,j]
                if not mask[i,j]:
                    ax.text(j,i,f'{val:.2f}',va='center',ha='center',color='black',fontsize=14)
                else:
                    ax.text(j,i,'x',va='center',ha='center',color='gray',alpha=0.5)
        ax.set_xticks(np.arange(n_cols))
        ax.set_yticks(np.arange(n_rows))
        ax.set_xticklabels(y_labels,rotation=45,ha='right',fontsize=12)
        ax.set_yticklabels(x_labels,fontsize=12)
        ax.set_title('Pearson Correlation Matrix', fontsize=18)
        fig.colorbar(cax, ax=ax, shrink=0.75, label='r')
        plt.tight_layout()
        fpath = os.path.join(self.plot_dir, 'corr_udips_128_256.png')
        plt.savefig(fpath)
    
    def plot_feature_target_correlations(self, fpath):
        """Plot feature correlations on train data- continuous variables only"""
        X = self.dataset.xtr
        Y = self.dataset.ytr[:,5:]
        feature_names = [f'UDIP {i}' for i in range(128)]
        target_names = self.covars
        corr_matrix = np.zeros((X.shape[1], Y.shape[1]))
        pvals = np.zeros_like(corr_matrix)
        for i in range(X.shape[1]):
            for j in range(Y.shape[1]):
                corr, pval = pearsonr(X[:, i], Y[:, j])
                corr_matrix[i, j] = corr
                pvals[i, j] = pval
        fig, ax = plt.subplots(figsize=(10, 6))
        cax = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        # Label axes
        ax.set_xticks(np.arange(len(target_names)))
        ax.set_yticks(np.arange(len(feature_names)))
        ax.set_xticklabels(target_names, fontsize=8, rotation=45, ha='right')
        ax.set_yticklabels(feature_names, fontsize=4)
        # Annotate each cell with the correlation coefficient
        #for i in range(len(feature_names)):
        #    for j in range(len(target_names)):
        #        text = f"{corr_matrix[i, j]:.2f}"
        #        ax.text(j, i, text, ha="center", va="center", color="black", fontsize=5)
        fig.colorbar(cax, ax=ax, label='Pearson r')
        ax.set_title("Feature-Target Correlations")
        plt.tight_layout()
        #plt.show()
        fpath = os.path.join(self.plot_dir, fpath)
        plt.savefig(fpath)
        plt.close()

    def plot_feature_target_dependencies(self, fpath, top_n=20):
        """Plot feature relevance scores on train data- categorical variables only"""
        x = self.dataset.xtr
        y = self.dataset.ytr[:,:5]
        feature_names = [f'UDIP {i}' for i in range(128)]
        n_targets = y.shape[1]
        target_names = self.covars_categorical
        fig, axes = plt.subplots(n_targets, 1, figsize=(12,3*n_targets), constrained_layout=True)
        for i in range(n_targets):
            yi = y[:,i]
            scores = mutual_info_classif(x, yi, discrete_features=False, random_state=0)
            idx_sorted = np.argsort(scores)[-top_n:]
            scores_top = scores[idx_sorted]
            features_top = np.array(feature_names)[idx_sorted]
            ax = axes[i]
            ax.bar(features_top, scores_top, color='teal')
            #ax.set_xticks(range(features_top))
            ax.set_xticklabels(features_top, rotation=90, fontsize=8)
            ax.set_ylabel('Mutual information score')
            ax.set_title(f'Mutual information with {target_names[i]}')
        fig.suptitle(f'Top {top_n} features for each target by mutual information')
        plt.tight_layout()
        #plt.show()
        fpath = os.path.join(self.plot_dir, fpath)
        plt.savefig(fpath)
        plt.close()

    def pls_regression(self, target_idx, n_comp=4):    
        """Run PLS regression on each continuous target (covariate)."""
        y_train = self.dataset.ytr[:,target_idx]
        y_test = self.dataset.yte[:,target_idx]
        pls = PLSRegression(n_components=n_comp)
        pls.fit(self.dataset.xtr, y_train)
        y_pred = pls.predict(self.dataset.xte)
        r2 = skm.r2_score(y_test, y_pred)
        rmse = np.sqrt(skm.mean_squared_error(y_test, y_pred))
        print(f"R² score (test): {r2:.3f}")
        print(f"RMSE (test): {rmse:.3f}")
        return y_pred, pls

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
    
    def plot_pls_predictions(self, y_true, y_pred, target, fpath):
        """Plot PLS test predictions vs true for continuous targets"""
        Y_true = np.asarray(y_true)
        Y_pred = np.asarray(y_pred)
        Y_true = Y_true.reshape(-1, 1)
        Y_pred = Y_pred.reshape(-1, 1)
        fig, ax = plt.subplots(1, 1, figsize=(5,4), squeeze=False)
        ax[0,0].scatter(Y_true, Y_pred, alpha=0.7, edgecolors='k')
        ax[0,0].plot([Y_true.min(), Y_true.max()], [Y_true.min(), Y_true.max()], 'r--', lw=2)
        ax[0,0].set_xlabel(f"True {target}")
        ax[0,0].set_ylabel(f"Predicted {target}")
        ax[0,0].set_title(f"PLS performance on {target}")
        ax[0,0].grid(True)
        plt.tight_layout()
        fpath = os.path.join(self.plot_dir, fpath)
        plt.savefig(fpath)
        plt.close()

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
    
    def calculate_vip(self, pls, target_idx):
        y = self.dataset.yte[:,target_idx]
        t = pls.x_scores_
        w = pls.x_weights_
        q = pls.y_loadings_
        p, h = w.shape
        vip = np.zeros((p,))
        ssy = np.sum((y - np.mean(y)) ** 2)
        for i in range(p):
            weight_sum = 0
            for comp in range(h):
                t_comp = t[:,comp]
                q_comp = q[0,comp]
                contrib = (t_comp ** 2).sum() * (q_comp ** 2)
                weight_sum += (w[i,comp] ** 2) * contrib
            vip[i] = np.sqrt(p * weight_sum / ssy)    
        return vip

    def plot_pls_vip_scores(self, pls, target_idx, fpath, top_n=10):
        """Plot top VIP scores from PLS regression on train data"""
        vip_scores = self.calculate_vip(pls, target_idx)
        feature_names = [f'UDIP {i}' for i in range(len(vip_scores))]
        sorted_idx = np.argsort(vip_scores)[::-1]
        top_idx = sorted_idx[:top_n]
        top_vip_scores = vip_scores[top_idx]
        top_feature_names = np.array(feature_names)[top_idx]
        plt.figure(figsize=(10, 5))
        bars = plt.bar(range(top_n), top_vip_scores, tick_label=top_feature_names)
        plt.axhline(1.0, color='r', linestyle='--', label='VIP = 1.0')
        plt.xlabel("Feature", fontsize=10)
        plt.ylabel("VIP Score", fontsize=10)
        plt.title(f"Top {top_n} Variable Importance in Projection (VIP) Features for target {target_idx}", fontsize=12)
        plt.xticks(rotation=45, fontsize=9)
        plt.yticks(fontsize=9)
        plt.legend()
        plt.tight_layout()
        #plt.show()
        fpath = os.path.join(self.plot_dir, fpath)
        plt.savefig(fpath)
        plt.close()
        important_features = np.array(feature_names)[vip_scores >= 1.0]
        #print(f"Important features (VIP ≥ 1): {important_features}")

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
    model = PredictiveModel(latent_size)

    # run correlation between 128-dim and 256-dim UDIP encodings
    #corrs, pvals = model.pearson_correlation()
    #model.plot_corr(corrs, pvals)

    # run LLE and plot
    x_lin = model.local_lin_embed(n_neighbors=500, n_components=3)
    model.plot_latent_space(x_lin, 'lle_apoe.png', 'LLE and APOE4 (# alleles)', labels=model.labels[:,1])
    model.plot_latent_space(x_lin, 'lle_cdr.png', 'LE and LogCDRSB', labels=model.labels[:,5])
    model.plot_latent_space(x_lin, 'lle_age.png', 'LLE and Age', labels=model.labels[:,4])
    model.plot_latent_space(x_lin, 'lle_sex.png', 'LLE and Sex (1=M, 2=F)', labels=model.labels[:,0])
    model.plot_latent_space(x_lin, 'lle_amyloid.png', 'LLE and Amyloid (0=neg, 1=pos)', labels=model.labels[:,3])
    model.plot_latent_space(x_lin, 'lle_dx.png', 'LLE and Diagnosis (1=CN, 2=MCI, 3=AD)', labels=model.labels[:,2])
    model.plot_latent_space(x_lin, 'lle_centiloids.png', 'LLE and Centiloid', labels=model.labels[:,6])
    model.plot_latent_space(x_lin, 'lle_wmh.png', 'LLE and LogNormWMH', labels=model.labels[:,7])
    model.plot_latent_space(x_lin, 'lle_gm.png', 'LLE and NormGM', labels=model.labels[:,8])
    model.plot_latent_space(x_lin, 'lle_wm.png', 'LLE and NormWM', labels=model.labels[:,9])
    model.plot_latent_space(x_lin, 'lle_csf.png', 'LLE and NormCSF', labels=model.labels[:,10])
    
    # run UMAP feature analysis and plot
    x_reduced = model.run_umap(n_neighbors=500, min_dist=0.1, n_components=3, metric='euclidean')
    model.plot_latent_space(x_reduced, 'umap_apoe.png', 'UMAP and APOE4 (# alleles)', labels=model.labels[:,1])
    model.plot_latent_space(x_reduced, 'umap_cdr.png', 'UMAP and LogCDRSB', labels=model.labels[:,5])
    model.plot_latent_space(x_reduced, 'umap_age.png', 'UMAP and Age', labels=model.labels[:,4])
    model.plot_latent_space(x_reduced, 'umap_sex.png', 'UMAP and Sex (1=M, 2=F)', labels=model.labels[:,0])
    model.plot_latent_space(x_reduced, 'umap_amyloid.png', 'UMAP and Amyloid (0=neg, 1=pos)', labels=model.labels[:,3])
    model.plot_latent_space(x_reduced, 'umap_dx.png', 'UMAP and Diagnosis (1=CN, 2=MCI, 3=AD)', labels=model.labels[:,2])
    model.plot_latent_space(x_reduced, 'umap_centiloids.png', 'UMAP and Centiloid', labels=model.labels[:,6])
    model.plot_latent_space(x_reduced, 'umap_wmh.png', 'UMAP and LogNormWMH', labels=model.labels[:,7])
    model.plot_latent_space(x_reduced, 'umap_gm.png', 'UMAP and NormGM', labels=model.labels[:,8])
    model.plot_latent_space(x_reduced, 'umap_wm.png', 'UMAP and NormWM', labels=model.labels[:,9])
    model.plot_latent_space(x_reduced, 'umap_csf.png', 'UMAP and NormCSF', labels=model.labels[:,10])


