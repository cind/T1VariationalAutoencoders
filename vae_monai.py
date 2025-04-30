# imports
import os, gc, random
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

# DL imports
import torch
from torch import nn
import torch.utils.checkpoint as cp
from torch.nn import functional as F
from torch.nn import MSELoss, KLDivLoss
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
from monai.losses.ssim_loss import SSIMLoss

# local imports
from dataset import *

# resizes images for loss function computation
def center_crop_to_shape(tensor, target_shape):
    assert tensor.ndim == len(target_shape), "input and target must have same number of dims"
    slices = []
    for i in range(tensor.ndim):
        diff = tensor.shape[i] - target_shape[i]
        if diff == 0:
            slices.append(slice(None))
        else:
            start = diff//2
            end = start + target_shape[i]
            slices.append(slice(start, end))
    return tensor[tuple(slices)]        


# Model architecture and forward pass to Pytorch lightning module.

class VAE(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.latent_size = 128
        self.input_shape = (1, 182, 218, 182)
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
                dropout=self.dropout_rate
                )
    
        self.loss_func = MSELoss(reduction="sum") + KLDivLoss(reduction="sum")

    def setup(self, stage: str):
        if hasattr(self.trainer.strategy, "_set_static_graph"):
            self.trainer.strategy._set_static_graph()
    
    def forward(self, x):
        return self.vae_model(x)
    
    # load from checkpoint
    def load_saved_model(self, checkpoint_path):
        model = VAE.load_from_checkpoint(checkpoint_path)    
        model.eval()
        model.freeze()
        return model
    
    # pytorch lightning training step
    def training_step(self, batch, batch_idx):
        # x, reg_input = batch
        x, mask = batch
        recon, mu, logvar, z = self(x)
        if recon.shape != x.shape:
            recon = center_crop_to_shape(recon, x.shape)
        loss1 = self.loss_func(x, recon)
        loss1 = loss1.squeeze(1) * mask
        loss1 = loss1.sum()
        loss1 = loss1 / mask.sum()
        # loss2 = self.train_loss_function(reg_input, reg)
        # loss = loss1 + loss2
        self.log("train_loss", loss1, prog_bar=True)
        return loss1

    # pytorch lightning validation step
    def validation_step(self, batch, batch_idx):
        x, mask = batch
        recon, mu, logvar, z = self(x)
        if recon.shape != x.shape:
            recon = center_crop_to_shape(recon, x.shape)
        loss1 = self.loss_func(x, recon)
        loss1 = loss1.squeeze(1) * mask
        loss1 = loss1.sum()
        loss1 = loss1 / mask.sum()
        # loss2 = self.train_loss_function(reg_input, reg)
        # loss = loss1 + loss2
        self.log("val_loss", loss1, prog_bar=True, sync_dist=True)
        return loss1

    # pytorch lightning test step
    def test_step(self, batch, batch_idx):
        x, mask = batch
        recon, mu, logvar, z = self(x)
        if recon.shape != x.shape:
            recon = center_crop_to_shape(recon, x.shape)
        loss1 = self.loss_func(x, recon)
        loss1 = loss1.squeeze(1) * mask
        loss1 = loss1.sum()
        loss1 = loss1 / mask.sum()
        # loss2 = self.train_loss_function(reg_input, reg)
        # loss = loss1 + loss2
        self.log("test_loss", loss1, prog_bar=True, sync_dist=True)
        return loss1
    
    # run inference on test set
    def test_model(self, model, dataloader):
        preds = []
        latents = []
        for batch in dataloader:
            x, mask = batch
            x = x.to(model.device)
            with torch.no_grad():
                recon, latent = model(x)
                recon = recon * mask
            preds.append(recon.cpu())
            latents.append(latent.cpu())
        return preds, latents    
    
    # visualize reconstructions of random sample of test data
    def plot_recon(self, model, test_loader, plot_title, device="cuda"):
        model.eval()
        model.to(device)
        test_dataset = test_loader.dataset
        indices = random.sample(range(len(test_dataset)), 10)
        orig_imgs = []
        recon_imgs = []
        # reconstruct sample data
        for idx in indices:
            x, mask = test_dataset[idx]
            mask = (mask > 0.5).float()
            x = x.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            with torch.no_grad():
                recon, mu, logvar, z = model(x)
                x = x * mask
                if recon.shape != x.shape:
                    recon = center_crop_to_shape(recon, x.shape)
                recon = recon * mask
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
        plt.savefig(plot_title)
    
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
train_dataset = aedataset(datafile="/m/Researchers/Eliana/DeepENDO/training/iopaths/train_data.txt", transforms=transforms_monai)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=12, pin_memory=True, num_workers=4, shuffle=True)
val_dataset = aedataset(datafile="/m/Researchers/Eliana/DeepENDO/training/iopaths/val_data.txt", transforms=transforms_monai)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=12, pin_memory=True, num_workers=4, shuffle=False)
test_dataset = aedataset(datafile="/m/Researchers/Eliana/DeepENDO/training/iopaths/test_data.txt", transforms=transforms_monai)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=12, pin_memory=True, num_workers=4, shuffle=True)

dir_name = "model_checkpoints"
model = VAE(0.001)
lr_monitor = LearningRateMonitor(logging_interval="epoch")

# saving checkpoints monitoring validation loss
model_checkpoint = ModelCheckpoint(
    dirpath=dir_name,
    monitor="val_loss",
    save_last=True,
    filename="{epoch}-{train_loss:.6f}-{val_loss:.6f}",
    save_top_k=5,
)

# loggers
#tb_logger = TensorBoardLogger(save_dir=dir_name + "/tb_logs")
csv_logger = CSVLogger(save_dir=dir_name + "/csv_logs")
pb = TQDMProgressBar()

# execute model training and testing
if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    
    trainer = pl.Trainer(
        logger=[csv_logger],
        precision="16-mixed",
        accelerator="gpu",
        devices=[0, 1, 2, 3],
        callbacks=[lr_monitor, model_checkpoint, pb],
        strategy="ddp_find_unused_parameters_true",
        detect_anomaly=False,
        sync_batchnorm=True,
        log_every_n_steps=20,
        benchmark=True,
        max_epochs=50,
    )

    # train model 
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # load trained model from checkpoint and test
    
    model_adni = model.load_saved_model(checkpoint_path="model_checkpoints/last.ckpt")
    print("Evaluating on test data...")
    test_rslt = trainer.test(
        model=model_adni,
        dataloaders=test_dataloader,
        verbose=True
    )
    print(f"Test loss: {test_rslt[0]['test_loss']:.6f}")

    # plot reconstructions from test data
    title = 'orig_recon_trainVAE_adni.png'
    model_adni.plot_recon(model=model_adni, test_loader=test_dataloader, plot_title=title)

