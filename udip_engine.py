# imports
import os, gc, random
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as get_psnr
from skimage.metrics import structural_similarity as get_ssim
from monai import transforms

# PyTorch
import torch
import torch.utils.checkpoint as cp
from torch.nn import functional as F
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ProgressBar,
    TQDMProgressBar
)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.functional import peak_signal_noise_ratio

# Custom imports
from dataset import *

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


# combined reconstruction (MSE/MAE) & SSIM loss
class ReconSSIMLoss(torch.nn.Module):
    def __init__(self, alpha, beta, loss_type='MSE', kernel_size=5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.kernel_size = kernel_size
        if loss_type == 'MSE':
            self.loss = torch.nn.MSELoss(reduction='none')
        elif loss_type == 'MAE':    
            self.loss = torch.nn.L1Loss(reduction='none')
        #self.data_range = data_range

    def forward(self, target: torch.Tensor, preds: torch.Tensor, mask: torch.Tensor):
        with torch.no_grad():
            mean_ssim = structural_similarity_index_measure(
                preds=preds,
                target=target,
                reduction='elementwise_mean',
                #data_range=self.data_range,
                kernel_size=self.kernel_size)
        loss = self.loss(target, preds)
        loss_masked = (loss.squeeze(1) * mask).sum() / mask.sum()
        total_loss = self.alpha * loss_masked + self.beta * (1 - mean_ssim)
        return total_loss


# combined MAE & PSNR loss
class MAE_PSNR_loss(torch.nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = torch.nn.MSELoss(reduction='none')
        self.mae_loss = torch.nn.L1Loss(reduction='none')

    def forward(self, target: torch.Tensor, preds: torch.Tensor, mask: torch.Tensor):
        mae = self.mae_loss(target, preds)
        mae_masked = (mae.squeeze(1) * mask).sum() / mask.sum()
        mse = self.mse_loss(target, preds)
        mse_masked = (mse.squeeze(1) * mask).sum() / mask.sum()
        psnr = 10 * torch.log10((target.max())**2 / (mse_masked + 1e-8))
        psnr_loss = 1.0 / (psnr + 1e-8)
        loss = self.alpha * mae_masked + self.beta * psnr_loss
        return loss


# Model architecture and forward pass to Pytorch lightning module.
class engine_AE(pl.LightningModule):
    def __init__(self, lr, latent_size=128, dropout_rate=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.latent_size = latent_size
        self.hidden_dim = latent_size
        self.dropout_rate = dropout_rate

        # defining layers
        # first CNN block
        self.first_cnn = self.first_CNN_block(1, 16)

        # encoders
        self.first_max_poold = self.max_poold((1, 1, 1))
        self.first_encoder = self.encoder_block(16, 32)
        self.second_max_poold = self.max_poold((0, 1, 0))
        self.second_encoder = self.encoder_block(32, 64)
        self.third_max_poold = self.max_poold((1, 0, 1))
        self.third_encoder = self.encoder_block(64, 128)
        self.fourth_max_poold = self.max_poold((0, 0, 0))
        self.fourth_encoder = self.encoder_block(128, 256)

        # latent space
        self.encoding_mlp = torch.nn.Linear(256 * 12 * 14 * 12, self.hidden_dim)

        self.decoding_mlp = torch.nn.Linear(self.hidden_dim, 256 * 12 * 14 * 12)

        # decoders
        self.first_decoder = self.decoder_block(256, 128)
        self.first_transconv = self.conv_transpose(128, input_padding=(0, 0, 0))
        self.second_decoder = self.decoder_block(128, 64)
        self.second_transconv = self.conv_transpose(64, input_padding=(1, 0, 1))
        self.third_decoder = self.decoder_block(64, 32)
        self.third_transconv = self.conv_transpose(32, input_padding=(0, 1, 0))
        self.fourth_decoder = self.decoder_block(32, 16)
        self.fourth_transconv = self.conv_transpose(16, input_padding=(1, 1, 1))

        # last CNN block
        self.last_cnn = self.last_CNN_block(16, 1)

        # MSE loss function
        self.mse_loss = torch.nn.MSELoss(reduction="none")

        # MAE loss function
        self.mae_loss = torch.nn.L1Loss(reduction='none')

        # loss functions combining metrics
        self.loss_func = ReconSSIMLoss(alpha=1, beta=1, loss_type='MSE')

    def max_poold(self, max_padding):
        max_pd = nn.MaxPool3d(kernel_size=2, padding=max_padding)
        return max_pd

    def encoder_block(self, input_channels, output_channels, padding=1):
        encoder = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, kernel_size=3, padding=padding,),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(inplace=True),
            #nn.Dropout3d(self.dropout_rate),
            nn.Conv3d(output_channels, output_channels, kernel_size=3, padding=padding),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(inplace=True),
        )

        return encoder

    def conv_transpose(self, output_channels, input_padding):
        conv_t = nn.ConvTranspose3d(
            output_channels,
            output_channels,
            kernel_size=2,
            stride=2,
            padding=input_padding,
        )

        return conv_t

    def decoder_block(self, input_channels, output_channels, input_padding=(0, 0, 0)):
        decoder = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, kernel_size=3, padding=1,),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(inplace=True),
            #nn.Dropout3d(self.dropout_rate),
            nn.Conv3d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(inplace=True),
        )
        return decoder

    def first_CNN_block(self, input_channels, output_channels, padding=1):
        cnn_block = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, kernel_size=3, padding=padding,),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(output_channels, output_channels, kernel_size=3, padding=padding),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(inplace=True),
        )

        return cnn_block

    def last_CNN_block(self, input_channels, output_channels, padding=1):
        cnn_block = nn.Sequential(
            nn.Conv3d(input_channels, input_channels, kernel_size=3, padding=padding),
            nn.BatchNorm3d(input_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(input_channels, input_channels, kernel_size=3, padding=padding),
            nn.BatchNorm3d(input_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(input_channels, output_channels, kernel_size=1),
        )

        return cnn_block

    def setup(self, stage: str):
        if hasattr(self.trainer.strategy, "_set_static_graph"):
            self.trainer.strategy._set_static_graph()
    
    # load from checkpoint
    def load_saved_model(self, checkpoint_path, eval_mode=True):
        model = engine_AE.load_from_checkpoint(checkpoint_path)    
        if eval_mode:
            model.eval()
            model.freeze()
        return model
    
    # Forward function
    def forward(self, x):
        #x = self.first_cnn(x)  # 1,16,182,218,182
        x = cp.checkpoint(self.first_cnn, x)
        x = self.first_max_poold(x)  # 1,16,92,110,92
        #x = self.first_encoder(x)  # 1,32,92,110,92
        x = cp.checkpoint(self.first_encoder, x)
        x = self.second_max_poold(x)  # 1,32,46,56,46
        #x = self.second_encoder(x)  # 1,64,46,56,46
        x = cp.checkpoint(self.second_encoder, x)
        x = self.third_max_poold(x)  # 1,64,24,28,24
        #x = self.third_encoder(x)  # 1,128,24,28,24
        x = cp.checkpoint(self.third_encoder, x)
        x = self.fourth_max_poold(x)  # 1,128,12,14,12
        x = self.fourth_encoder(x)  # 1,256,12,14,12
        #x = cp.checkpoint(self.fourth_encoder, x)
        shape = x.size()

        # flattening encoder output to keep batch dimension intact
        enc_features = torch.flatten(x, start_dim=1, end_dim=-1)

        lin1 = self.encoding_mlp(enc_features)  # 1,128
        # Going from hidden dimension to original image recon
        dec = self.decoding_mlp(lin1)  # 1,516096
        dec = dec.view(shape)  # 1,256,12,14,12
        dec = self.first_decoder(dec)  # 1,128,12,14,12
        #dec = cp.checkpoint(self.first_decoder, dec)
        dec = self.first_transconv(dec)  # 1,128,24,28,24
        dec = self.second_decoder(dec)  # 1,64,24,28,24
        #dec = cp.checkpoint(self.second_decoder, dec)
        dec = self.second_transconv(dec)  # 1,64,46,56,46
        dec = self.third_decoder(dec)  # 1,32,46,56,46
        #dec = cp.checkpoint(self.third_decoder, dec)
        dec = self.third_transconv(dec)  # 1,32,92,110,92
        dec = self.fourth_decoder(dec)  # 1,16,92,110,92
        #dec = cp.checkpoint(self.fourth_decoder, dec)
        dec = self.fourth_transconv(dec)  # 1,16,182,218,182
        recon = self.last_cnn(dec)  # 1, 182, 218, 182
        #recon = cp.checkpoint(self.last_cnn, dec)

        return recon, lin1

    # pytorch lightning training step
    def training_step(self, batch, batch_idx):
        x, mask = batch
        recon, _ = self(x)
        #loss1 = self.mse_loss(x, recon)
        #loss1 = self.mae_loss(x, recon)
        #loss1 = loss1.squeeze(1) * mask
        #loss1 = loss1.sum()
        #loss1 = loss1 / mask.sum()
        loss1 = self.loss_func(x, recon, mask)
        self.log("train_loss", loss1, prog_bar=True)
        return loss1

    # pytorch lightning validation step
    def validation_step(self, batch, batch_idx):
        x, mask = batch
        recon, _ = self(x)
        #loss1 = self.mse_loss(x, recon)
        #loss1 = self.mae_loss(x, recon)
        #loss1 = loss1.squeeze(1) * mask
        #loss1 = loss1.sum()
        #loss1 = loss1 / mask.sum()
        loss1 = self.loss_func(x, recon, mask)
        self.log("val_loss", loss1, prog_bar=True, sync_dist=True)
        return loss1

    # pytorch lightning test step
    def test_step(self, batch, batch_idx):
        x, mask = batch
        recon, _ = self(x)
        #loss1 = self.mse_loss(x, recon)
        #loss1 = self.mae_loss(x, recon)
        #loss1 = loss1.squeeze(1) * mask
        #loss1 = loss1.sum()
        #loss1 = loss1 / mask.sum()
        loss1 = self.loss_func(x, recon, mask)
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
    
    # visualize reconstructed data and output losses
    def plot_recon(self, model, loader, plot_title, csv_title, device="cuda"):
        model.eval()
        model.to(device)
        test_dataset = loader.dataset
        indices = random.sample(range(len(test_dataset)), 10)
        orig_imgs = []
        recon_imgs = []
        df = pd.DataFrame(columns=['ImgCode','MSE','MAE','PSNR','SSIM'])
        # reconstruct sample data
        for idx in indices:
            x, mask, img_name = test_dataset[idx]
            #x, mask = test_dataset[idx]
            mask = (mask > 0.5).float()
            x = x.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            with torch.no_grad():
                print(f'reconstructing image {idx}')
                recon, _ = model(x)
                #print(x.shape, recon.shape, mask.shape)
                x = x * mask
                recon = recon * mask
                mse, mae, psnr_val, ssim_val = compute_metrics(x, recon, mask)
                df.loc[len(df)] = [img_name, mse, mae, psnr_val, ssim_val]
            orig_imgs.append(x.cpu())
            recon_imgs.append(recon.cpu())
            # save orig and recon niftis
            #orig_nii = nib.Nifti1Image(x.squeeze().cpu().numpy(), aff)
            #orig_nii.to_filename(f'orig_{img_name}')
            #recon_nii = nib.Nifti1Image(recon.squeeze().cpu().numpy(), aff)
            #recon_nii.to_filename(f'recon_{img_name}')
        orig_imgs = torch.cat(orig_imgs, dim=0)
        recon_imgs = torch.cat(recon_imgs, dim=0)
        df.to_csv(csv_title, index=False)
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
    def configure_optimizers(self, weight_decay_rate=1e-6):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"], weight_decay=weight_decay_rate)
        lr_scheduler_config = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                "min",
                patience=10,
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

        
# defining train dataset
train_dataset = aedataset(
    datafile="/m/Researchers/Eliana/DeepENDO/training/iopaths/adni_dlmuse_normative/train_paths.txt", 
    transforms=transforms_monai,
    #return_affine=True,
    return_img_name=True
)

# defining train dataloader
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=12, pin_memory=True, num_workers=4, shuffle=True,
)

# defining validation dataset
val_dataset = aedataset(
    datafile="/m/Researchers/Eliana/DeepENDO/training/iopaths/adni_dlmuse_normative/val_paths.txt",
    transforms=transforms_monai,
    #return_affine=True,
    return_img_name=True
)

# defining validation dataloader
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=12, pin_memory=True, num_workers=4, shuffle=False
)

# defining test dataset
test_dataset = aedataset(
        datafile="/m/Researchers/Eliana/DeepENDO/training/iopaths/adni_dlmuse_normative/test_paths.txt",
        transforms=transforms_monai,
        #return_affine=True,
        return_img_name=True
)

# defining test dataloader
test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, pin_memory=True, num_workers=4, shuffle=True
)
'''
# defining discovery dataset
disc_dataset = aedataset(
        datafile="/m/Researchers/Eliana/DeepENDO/training/iopaths/adni_init/discovery_data.txt",
        transforms=transforms_monai,
        return_affine=True,
        return_img_name=True
)

# defining discovery dataloader
disc_dataloader = torch.utils.data.DataLoader(
        disc_dataset, batch_size=12, pin_memory=True, num_workers=4, shuffle=True
)
'''
# directory name to save checkpoints and metrics
dir_name = '/m/Researchers/Eliana/DeepENDO/training/T1_128/DLMUSE_ADNI_normative'

# initiaing the model
latent_size = 128
orig_lr = 0.0005248074602497723
lr = 0.001
AE_model = engine_AE(lr, latent_size)

# learning rate monitor as using scheduler
lr_monitor = LearningRateMonitor(logging_interval="epoch")

# early stopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=40)

# saving checkpoints monitoring validation loss
model_checkpoint = ModelCheckpoint(
    dirpath=dir_name,
    monitor="val_loss",
    save_last=True,
    filename="{epoch}-{train_loss:.4f}-{val_loss:.4f}",
    save_top_k=8,
)

# Loggers
#tb_logger = TensorBoardLogger(save_dir=dir_name + "/tb_logs")
#csv_logger = CSVLogger(save_dir=dir_name + "/csv_logs")
pb = TQDMProgressBar()

# main training
if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    
    trainer = pl.Trainer(
        #logger=[tb_logger, csv_logger],
        precision="16-mixed",
        # Change the number of GPUs here
        accelerator="gpu",
        devices=[0, 1, 2, 3],
        #devices=[0, 1],
        callbacks=[lr_monitor, model_checkpoint, pb, early_stop],
        strategy="ddp_find_unused_parameters_true",
        detect_anomaly=False,
        sync_batchnorm=True,
        log_every_n_steps=20,
        benchmark=True,
        max_epochs=100,
    )

    #orig_ckpt = '/m/Researchers/Eliana/DeepENDO/UDIP/ckpts/T1.ckpt'
    trained_ckpt = '/m/Researchers/Eliana/DeepENDO/training/T1_128/DLMUSE_ADNI_normative/last.ckpt'
    
    # load original UDIP checkpoint in training mode
    #AE_model = AE_model.load_saved_model(checkpoint_path=orig_ckpt, eval_mode=False)
    
    # train model 
    #trainer.fit(AE_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # load trained model from checkpoint
    AE_model = AE_model.load_saved_model(checkpoint_path=trained_ckpt, eval_mode=True)
    
    # test model
    #print("Evaluating on test data...")
    #test_rslt = trainer.test(model=AE_model, dataloaders=test_dataloader, verbose=True)
    #print(f"Test loss: {test_rslt[0]['test_loss']:.6f}")
    
    # compute losses and plot reconstructions
    plot_train = 'adni_dlmuse_normative_traindata_udip_recon.png'
    csv_train = 'adni_dlmuse_normative_traindata_udip_losses.csv'
    plot_val = 'adni_dlmuse_normative_valdata_udip_recon.png'
    csv_val = 'adni_dlmuse_normative_valdata_udip_losses.csv'
    plot_test = 'adni_dlmuse_normative_testdata_udip_recon.png'
    csv_test = 'adni_dlmuse_normative_testdata_udip_losses.csv'
    #plot_disc = 'adni_dlmuse_normative_discdata_udip_recon.png'
    #csv_disc = 'adni_dlmuse_normative_discdata_udip_losses.csv'
    AE_model.plot_recon(model=AE_model, loader=train_dataloader, plot_title=plot_train, csv_title=csv_train)
    AE_model.plot_recon(model=AE_model, loader=val_dataloader, plot_title=plot_val, csv_title=csv_val)
    AE_model.plot_recon(model=AE_model, loader=test_dataloader, plot_title=plot_test, csv_title=csv_test)
    #AE_model.plot_recon(model=AE_model, loader=disc_dataloader, plot_title=plot_disc, csv_title=csv_disc)

