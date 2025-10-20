import os, math, gc, random
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as get_psnr
from skimage.metrics import structural_similarity as get_ssim

# DL imports
import torch
from torch import nn
import torch.utils.checkpoint as cp
from torch.nn import functional as F
from torch.nn import MSELoss, L1Loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ProgressBar,
    TQDMProgressBar)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.functional import peak_signal_noise_ratio

import monai
from monai.networks.nets import ViT

# local imports
from dataset import *

# loss functions
def masked_mse(recon, x, mask):
    recon_loss = F.mse_loss(recon, x, reduction='none')
    recon_loss_masked = (recon_loss.squeeze(1)).sum() / mask.sum()
    return recon_loss_masked

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
    psnr_val = get_psnr(original, recon, data_range=original.max() - original.min())
    ssim_val = get_ssim(original, recon, data_range=original.max() - original.min())
    return mse, mae, psnr_val, ssim_val

# save metrics to csv
def save_metrics(metrics_list, samples, out_csv):
    rows = []
    for sample_name, sample_metric in zip(samples, metrics_list):
        for metric in sample_metric:
            rows.append({"Sample": sample_name,
                         "MSE": mse,
                         "MAE": mae,
                         "PSNR": psnr_val,
                         "SSIM": ssim_val})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

# model architecture
class ViT(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.latent_size = 128
        self.patch_size = (16, 16, 16)
        self.input_shape = (176, 224, 176)
        self.dropout_rate = 0.0
        self.num_layers = 4
        self.mlp_dim = 512
        self.num_heads = 4
        self.vit_model = ViTAutoEnc(
                spatial_dims=3,
                in_channels=1,
                img_size=self.input_shape,
                patch_size=self.patch_size,
                hidden_size=self.latent_size,
                mlp_dim=self.mlp_dim,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                classification=False
                )
    
    def setup(self, stage: str):
        if hasattr(self.trainer.strategy, "_set_static_graph"):
            self.trainer.strategy._set_static_graph()
    
    def forward(self, x):
        return self.vit_model(x)
    
    # load model from checkpoint
    def load_saved_model(self, checkpoint_path):
        model = ViT.load_from_checkpoint(checkpoint_path)    
        model.eval()
        model.freeze()
        return model
    
    # pytorch lightning training step
    def training_step(self, batch, batch_idx):
        x, mask = batch
        recon, z = self(x)
        loss = masked_mse(recon, x, mask)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    # pytorch lightning validation step
    def validation_step(self, batch, batch_idx):
        x, mask = batch
        recon, z = self(x)
        loss = masked_mse(recon, x, mask)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    # pytorch lightning test step
    def test_step(self, batch, batch_idx):
        x, mask = batch
        recon, z = self(x)
        loss = masked_mse(recon, x, mask)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    # visualize reconstructions of random sample of test data
    def plot_recon(self, model, dataloader, plot_filepath, csv_filepath, device="cuda"):
        model.eval()
        model.to(device)
        dataset = dataloader.dataset
        indices = random.sample(range(len(dataset)), 10)
        orig_imgs = []
        recon_imgs = []
        df = pd.DataFrame(columns=['ImgIdx','MSE','MAE','PSNR','SSIM'])
        # reconstruct sample data
        for idx in indices:
            x, mask = dataset[idx]
            #mask = (mask > 0.5).float()
            x = x.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            with torch.no_grad():
                x = x * mask
                recon, z = model(x)
                recon = recon * mask
                mse, mae, psnr_val, ssim_val = compute_metrics(x, recon, mask)
                df.loc[len(df)] = [idx, mse, mae, psnr_val, ssim_val]
            orig_imgs.append(x.cpu())
            recon_imgs.append(recon.cpu())
        orig_imgs = torch.cat(orig_imgs, dim=0)
        recon_imgs = torch.cat(recon_imgs, dim=0)
        df.to_csv(csv_filepath, index=False)
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
            "monitor": "val_loss",
            "strict": True,
            "name": None,
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }

# defining datasets
iodir = '/m/Researchers/Eliana/DeepENDO/training/iopaths'
train_dataset = aedataset(datafile=os.path.join(iodir, "adni_dlmuse_normative/train_paths.txt"))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=12, pin_memory=True, num_workers=4, shuffle=True)
val_dataset = aedataset(datafile=os.path.join(iodir, "adni_dlmuse_normative/val_paths.txt"))
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=12, pin_memory=True, num_workers=4, shuffle=False)
test_dataset = aedataset(datafile=os.path.join(iodir, "adni_dlmuse_normative/test_paths.txt"))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=12, pin_memory=True, num_workers=4, shuffle=True)

dir_name = "model_checkpoints/ViT"
model = ViT(lr=0.001)
lr_monitor = LearningRateMonitor(logging_interval="epoch")

# saving checkpoints monitoring validation loss
model_checkpoint = ModelCheckpoint(
    dirpath=dir_name,
    monitor="val_loss",
    save_last=True,
    filename="{epoch}-{train_loss:.4f}-{val_loss:.4f}",
    save_top_k=8)

# loggers
pb = TQDMProgressBar()

# execute model training and testing
if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    
    epochs=100
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

    vit_ckpt = "model_checkpoints/ViT/last.ckpt"
    
    # train model 
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # test model
    model0 = model
    #model0 = model.load_saved_model(checkpoint_path=vit_ckpt)
    
    print("Evaluating on test data")
    test_rslt = trainer.test(model=model0, dataloaders=test_dataloader, verbose=True)
    print(f"Test loss: {test_rslt[0]['test_loss']:.6f}")

    # save losses and plot reconstructions from each dataset
    plot_train = 'adni_dlmuse_normative_traindata_udip_recon.png'
    csv_train = 'adni_dlmuse_normative_traindata_udip_losses.csv'
    plot_val = 'adni_dlmuse_normative_valdata_udip_recon.png'
    csv_val = 'adni_dlmuse_normative_valdata_udip_losses.csv'
    plot_test = 'adni_dlmuse_normative_testdata_udip_recon.png'
    csv_test = 'adni_dlmuse_normative_testdata_udip_losses.csv'
    model0.plot_recon(model=model0, dataloader=train_dataloader, plot_filepath=plot_train, csv_filepath=csv_train)
    model0.plot_recon(model=model0, dataloader=val_dataloader, plot_filepath=plot_val, csv_filepath=csv_val)
    model0.plot_recon(model=model0, dataloader=test_dataloader, plot_filepath=plot_test, csv_filepath=csv_test)

