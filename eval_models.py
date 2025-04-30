import os
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from scipy.ndimage import zoom
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.nn import functional as F
from monai import transforms
# local imports
from dataset import aedataset 
from udip_engine import engine_AE


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

# plot original and reconstructed data
def plot_comparisons(originals, recons_list, model_names, samples, plot_path):
    n_samples = len(samples)
    n_models = len(model_names)
    fig = plt.figure(figsize=(4*(n_models+1), 3*n_samples))
    gs = gridspec.GridSpec(n_samples, n_models+1, figure=fig, wspace=0.05, hspace=0.3)

    for i in range(n_samples):
        ax = fig.add_subplot(gs[i,0])
        ax.imshow(originals[i], cmap='gray')
        ax.axis('off')
        if i==0:
            ax.set_title('Original', fontsize=12)
        ax.set_ylabel(os.path.basename(samples[i]), rotation=0, labelpad=40, va='center', fontsize=10)    
        for j in range(n_models):
            ax = fig.add_subplot(gs[i, j+1])
            ax.imshow(recons_list[j][i], cmap='gray')
            ax.axis('off')
            if i==0:
                ax.set_title(model_names[j], fontsize=12)
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()

# main function to compare models
def run_comparison(torch_paths, keras_path, datafile, transforms, out_dir, device="cuda"):
    sample_list = [line.replace('\n','') for line in open(datafile, 'r')]
    sample_dataset = aedataset(datafile=datafile, transforms=transforms)
    torch_models = []
    for path in torch_paths:
        model = engine_AE.load_from_checkpoint(path).to(device)
        model.eval()
        torch_models.append(model)
    keras_model = tf.keras.models.load_model(keras_path)    
    all_metrics = []
    originals = []
    all_recons = [[] for _ in range(len(torch_models)+1)]
    
    # run models on sample
    for idx in range(len(sample_dataset)):
        x, mask = sample_dataset[idx]
        x = x.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        original = x*mask
        recons = []
        metrics = []
        for model in torch_models:
            with torch.no_grad():
                recon, _ = model(x)
                recon = recon * mask
            recons.append(recon.squeeze().cpu().numpy())
            metrics.append(compute_metrics(original, recon, mask))
        x_np = x.squeeze().cpu().numpy()
        x_np = np.expand_dims(x_np, axis=(0,-1))
        target_shape = (180, 220, 180)
        current_shape = x_np.shape[1:4]
        zoom_factors = [t/c for t,c in zip(target_shape, current_shape)]
        x_np = zoom(x_np, zoom=[1]+zoom_factors+[1], order=1)
        recon_np = keras_model.predict(x_np)
        mask_np = mask.squeeze().cpu().numpy()
        mask_np = np.expand_dims(mask_np, axis=(0,-1))
        mask_np = zoom(mask_np, zoom=[1]+zoom_factors+[1], order=1)
        #recon_np_masked = recon_np * mask_np
        recons.append(recon_np)
        keras_metrics = compute_metrics(x_np.squeeze(), recon_np.squeeze())
        metrics.append(keras_metrics)
        mid_slice_idx = original.shape[-1]//2
        originals.append(original.squeeze().cpu().numpy()[:,:,mid_slice_idx])
        for model_idx, recon in enumerate(recons):
            all_recons[model_idx].append(recon.squeeze()[:,:,mid_slice_idx])
        all_metrics.append(metrics)    
    
    # save metrics and plot
    model_names = ['udip_pretrained', 'udip_ADNItrained', 'antsCAE_ADNItrained']
    save_metrics(all_metrics, model_names, sample_list, out_csv=f'{out_dir}/metrics.csv')
    plot_comparisons(
            originals=originals,
            recons_list=all_recons,
            model_names=model_names,
            samples=sample_list,
            plot_path=f'{out_dir}/model_comparison.png')

if __name__ == "__main__":
    #transforms_monai = transforms.Compose([transforms.EnsureChannelFirst(), transforms.ScaleIntensity(), transforms.Resize((182, 218, 182))])
    transforms_monai = transforms.Compose([transforms.AddChannel(), transforms.ToTensor()])

    torch_paths = ["/m/Researchers/Eliana/DeepENDO/UDIP/ckpts/T1.ckpt", 
                   "/m/Researchers/Eliana/DeepENDO/training/T1_128/trainAE_adni/last.ckpt"]
    keras_path = "saved_models/cae_fmap128_adni_data.keras"
    sample_datafile = "iopaths/sample_test_vis.txt"
    out_dir = "model_performance"

    run_comparison(torch_paths, keras_path, sample_datafile, transforms_monai, out_dir)






