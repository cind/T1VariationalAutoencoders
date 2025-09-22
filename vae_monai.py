# imports
import os, math, gc, random
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# DL imports
import torch
from torch import nn
import torch.utils.checkpoint as cp
from torch.nn import functional as F
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ProgressBar,
    TQDMProgressBar)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import monai
from monai.networks.nets import VarAutoEncoder
from monai import transforms

# local imports
from dataset import *

# loss functions
def vae_loss(recon, x, mu, logvar, beta, scale_mse=1):
    recon = torch.clamp(recon, min=-10, max=-10)
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    logvar = torch.clamp(logvar, min=-10, max=10)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = kl_loss.mean()
    loss = scale_mse*recon_loss + beta*kl_loss
    return loss, recon_loss, kl_loss

def cae_loss(recon, x, mask=None):
    if mask is not None:
        #print("recon min/max/mean/nan:", recon.min().item(), recon.max().item(), recon.mean().item(), torch.isnan(recon).sum().item())
        #print("mask sum/nan:", mask.sum().item(), torch.isnan(mask).sum().item())
        x = x * mask
        recon = torch.clamp(recon, 0.0, 1.0)
        recon = recon * mask
        se = (recon - x) ** 2
        vxls = mask.sum() + 1e-8
        loss = se.sum() / vxls
        #loss = F.mse_loss(recon, x, reduction="none")
        #loss = loss.squeeze(1) * mask
        #loss = loss.sum() / mask.sum()
    else:
        loss = F.mse_loss(recon, x, reduction="mean")
    return loss

# compute MSE, MAE, PSNR, SSIM from reconstructed sample
def compute_metrics(original, recon, mask=None):
    if isinstance(original, torch.Tensor):
        original = original.squeeze().cpu().numpy()
    if isinstance(recon, torch.Tensor):    
        recon = recon.squeeze().cpu().numpy()
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze().cpu().numpy()
        original *= mask
        recon *= mask
    mse = np.mean((original - recon) ** 2)
    mae = np.mean(np.abs(original - recon))
    psnr_val = psnr(original, recon, data_range=original.max() - original.min())
    ssim_val = ssim(original, recon, data_range=original.max() - original.min())
    return mse, mae, psnr_val, ssim_val

# save metrics to csv
def save_metrics(metrics_list, model_names, samples, out_csv):
    rows = []
    for sample_name, sample_metric in zip(samples, metrics_list):
        for model_name, (mse, mae, psnr_val, ssim_val) in zip(model_names, sample_metric):
            rows.append({"Sample": sample_name,
                         "Model": model_name,
                         "MSE": mse,
                         "MAE": mae,
                         "PSNR": psnr_val,
                         "SSIM": ssim_val})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

# model architecture
class VAE(pl.LightningModule):
    def __init__(self, lr, beta):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.beta = beta
        self.beta_max = 0.1
        self.anneal_epochs = 20
        self.latent_size = 128
        self.input_shape = (1, 184, 224, 184)
        self.dropout_rate = 0.2
        self.kernel_size = 5
        self.channels = (16, 32, 64)
        self.strides = (2, 2, 2)
        self.vae_model = VarAutoEncoder(
                spatial_dims=3,
                in_shape=self.input_shape,
                out_channels=1,
                latent_size=self.latent_size,
                channels=self.channels,
                strides=self.strides,
                kernel_size=self.kernel_size,
                up_kernel_size=self.kernel_size,
                dropout=self.dropout_rate,
                norm='BATCH')
    
    def setup(self, stage: str):
        if hasattr(self.trainer.strategy, "_set_static_graph"):
            self.trainer.strategy._set_static_graph()
    
    def forward(self, x):
        return self.vae_model(x)
    
    # load model from checkpoint
    def load_saved_model(self, checkpoint_path):
        model = VAE.load_from_checkpoint(checkpoint_path)    
        model.eval()
        model.freeze()
        return model
    
    # transfer encoder/decoder weights from CAE to VAE
    def transfer_weights(self, checkpoint_path):
        cae_ckpt = torch.load(checkpoint_path)
        cae_state = cae_ckpt["state_dict"] if "state_dict" in cae_ckpt else cae_ckpt
        vae_state = self.vae_model.state_dict()
        xferred_keys = []
        # transfer encoder and decoder weights 
        pfx = "vae_model."
        for cae_key, cae_weight in cae_state.items():
            k = cae_key.replace(pfx,"")
            if k in vae_state and cae_weight.shape == vae_state[k].shape:
                vae_state[k] = cae_weight
                xferred_keys.append(k)
        self.vae_model.load_state_dict(vae_state)        
        print(f"[INFO] Transferred {len(xferred_keys)} encoder & decoder layers to VAE")
        # re-initialize remaining layers
        xferred_pfxs = set(k.rsplit(".",1)[0] for k in xferred_keys)
        for name, module in self.vae_model.named_modules():
            if name in xferred_pfxs:
                continue
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
                if hasattr(module, "weight") and module.weight is not None:
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, (nn.BatchNorm3d, nn.InstanceNorm3d)): 
                if hasattr(module, "weight") and module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if hasattr(module, "bias") and module.bias is not None:    
                    nn.init.constant_(module.bias, 0.0)
        print(f"[INFO] Re-initialized non-transferred layers")
    
    # KL annealing (sigmoidal schedule) for model stability
    def kl_anneal(self, epoch, k=2, start=10, end=40):
        if epoch < start:
            return 0.0
        elif epoch > end:
            return self.beta_max
        else:
            progr = (epoch-start)/(end-start)
            b = self.beta_max/(1+np.exp(-k*(progr-0.5)))
            return b.item()
    
    # pytorch lightning training step
    def training_step(self, batch, batch_idx):
        x, mask = batch
        recon, mu, logvar, z = self(x)
        #recon, z = self(x)
        #loss = cae_loss(recon, x, mask)
        beta = self.kl_anneal(self.current_epoch)
        loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, beta)
        self.log("train_total_loss", loss, prog_bar=True)
        self.log("train_recon_loss", recon_loss, prog_bar=True)
        self.log("train_kl_loss", kl_loss, prog_bar=True)
        self.log("train_beta", beta, prog_bar=True)
        return loss

    # pytorch lightning validation step
    def validation_step(self, batch, batch_idx):
        x, mask = batch
        recon, mu, logvar, z = self(x)
        #recon, z = self(x)
        #loss = cae_loss(recon, x, mask)
        loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, self.beta)
        self.log("val_total_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_recon_loss", recon_loss, prog_bar=True, sync_dist=True)
        self.log("val_kl_loss", kl_loss, prog_bar=True, sync_dist=True)
        return loss

    # pytorch lightning test step
    def test_step(self, batch, batch_idx):
        x, mask = batch
        #recon, mu, logvar, z = self(x)
        recon, z = self(x)
        loss = cae_loss(recon, x, mask)
        #loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, self.beta_max)
        #self.log("test_total_loss", loss, prog_bar=True, sync_dist=True)
        #self.log("test_recon_loss", recon_loss, prog_bar=True, sync_dist=True)
        #self.log("test_kl_loss", kl_loss, prog_bar=True, sync_dist=True)
        return loss
    
    # visualize reconstructions of random sample of test data
    def plot_recon(self, model, test_loader, plot_filepath, device="cuda"):
        model.eval()
        model.to(device)
        test_dataset = test_loader.dataset
        indices = random.sample(range(len(test_dataset)), 10)
        orig_imgs = []
        recon_imgs = []
        # reconstruct sample data
        for idx in indices:
            x, mask = test_dataset[idx]
            #mask = (mask > 0.5).float()
            x = x.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            with torch.no_grad():
                x = x * mask
                #recon, mu, logvar, z = model(x)
                recon, z = model(x)
                recon = recon * mask
                mse, mae, psnr_val, ssim_val = compute_metrics(x, recon, mask)
                print(f"Sample {idx}: MSE={mse}, MAE={mae}, PSNR={psnr_val}, SSIM={ssim_val}")
            orig_imgs.append(x.cpu())
            recon_imgs.append(recon.cpu())
        orig_imgs = torch.cat(orig_imgs, dim=0)
        recon_imgs = torch.cat(recon_imgs, dim=0)
        # plot orig and recon
        fig, axes = plt.subplots(nrows=10, ncols=6, figsize=(15,25))
        slice_fracs = [0.25, 0.5, 0.75]
        for row in range(10):
            orig = orig_imgs[row, 0]
            recon = recon_imgs[row, 0]
            d = orig.shape[-1]
            slice_indices = [int(d*f) for f in slice_fracs]
            for i, idx in enumerate(slice_indices):
                axes[row, i].imshow(orig[:, :, idx], cmap='gray')
                axes[row, i].axis('off')
                if i == 0:
                    axes[row, i].set_title('Original')
                axes[row, i+3].imshow(recon[:, :, idx], cmap='gray')
                axes[row, i+3].axis('off')
                if i == 0:
                    axes[row, i+3].set_title('Reconstructed')
        plt.tight_layout()
        plt.savefig(plot_filepath)
    
    # pytorch lightning optimizer configuration
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
        lr_scheduler_config = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                "min",
                patience=4,
                min_lr=self.hparams["lr"] / 1000,
                factor=0.5,
            ),
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_total_loss",
            "strict": True,
            "name": None,
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }

# defining datasets
train_dataset = aedataset(datafile="/m/Researchers/Eliana/DeepENDO/training/iopaths/train_data.txt", transforms=transforms_monai)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=12, pin_memory=True, num_workers=4, shuffle=True)
val_dataset = aedataset(datafile="/m/Researchers/Eliana/DeepENDO/training/iopaths/val_data.txt", transforms=transforms_monai)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=12, pin_memory=True, num_workers=4, shuffle=False)
test_dataset = aedataset(datafile="/m/Researchers/Eliana/DeepENDO/training/iopaths/test_data.txt", transforms=transforms_monai)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=12, pin_memory=True, num_workers=4, shuffle=True)

dir_name = "model_checkpoints/vae/transfer"
model = VAE(lr=0.001, beta=0.1)
lr_monitor = LearningRateMonitor(logging_interval="epoch")

# saving checkpoints monitoring validation loss
model_checkpoint = ModelCheckpoint(
    dirpath=dir_name,
    monitor="val_total_loss",
    save_last=True,
    filename="{epoch}-{train_total_loss:.6f}-{val_total_loss:.6f}",
    save_top_k=5)

# loggers
pb = TQDMProgressBar()

# execute model training and testing
if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    
    epochs=50
    trainer = pl.Trainer(
        precision="16-mixed",
        accelerator="gpu",
        devices=[0, 1, 2, 3],
        callbacks=[lr_monitor, model_checkpoint, pb],
        strategy="ddp_find_unused_parameters_true",
        detect_anomaly=False,
        sync_batchnorm=True,
        gradient_clip_val=1.0,
        log_every_n_steps=20,
        benchmark=True,
        max_epochs=epochs,
    )

    # transfer encoder and decoder weights from trained CAE --> current VAE
    cae_ckpt = "model_checkpoints/cae/vae-noresample-noKL.ckpt"
    #model.transfer_weights(cae_ckpt)
    
    # train model 
    #trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # load trained model from checkpoint and test 
    model0 = model.load_saved_model(checkpoint_path=cae_ckpt)
    #model0=model
    
    print("Evaluating on test data")
    '''
    test_rslt = trainer.test(
        model=model0,
        dataloaders=test_dataloader,
        verbose=True
    )'''
    #print(f"Test loss: {test_rslt[0]['test_total_loss']:.6f}")

    # plot reconstructions from test data
    #p = 'model_performance/VAE_xfer_orig_recon_adni.png'
    p = 'model_performance/CAE_orig_recon_adni.png'
    model0.plot_recon(model=model0, test_loader=test_dataloader, plot_filepath=p)

