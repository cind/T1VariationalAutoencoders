import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.nn import functional as F
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, Resize
from engine_AE import engine_AE  # Your PyTorch autoencoder class
from dataset import aedataset   # Your dataset
import tensorflow as tf

def compute_metrics(original, recon, mask=None):
    original_np = original.squeeze().cpu().numpy()
    recon_np = recon.squeeze().cpu().numpy()
    if mask is not None:
        mask_np = mask.squeeze().cpu().numpy()
        original_np *= mask_np
        recon_np *= mask_np
    mse = np.mean((original_np - recon_np) ** 2)
    mae = np.mean(np.abs(original_np - recon_np))
    psnr_val = psnr(original_np, recon_np, data_range=original_np.max() - original_np.min())
    ssim_val = ssim(original_np, recon_np, data_range=original_np.max() - original_np.min())
    return mse, mae, psnr_val, ssim_val

def plot_comparisons(original, recons, model_names, title, slice_fracs=[0.25, 0.5, 0.75]):
    fig, axes = plt.subplots(nrows=len(recons) + 1, ncols=len(slice_fracs), figsize=(12, 10))
    d = original.shape[-1]
    slice_indices = [int(d * f) for f in slice_fracs]

    for col, idx in enumerate(slice_indices):
        axes[0, col].imshow(original[..., idx], cmap="gray")
        axes[0, col].set_title(f"Original - Slice {idx}")
        axes[0, col].axis("off")

    for row, (recon, name) in enumerate(zip(recons, model_names), start=1):
        for col, idx in enumerate(slice_indices):
            axes[row, col].imshow(recon[..., idx], cmap="gray")
            axes[row, col].set_title(name)
            axes[row, col].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.show()

def run_comparison(pt_paths, pt_names, keras_path, keras_name, datafile, transforms, device="cuda"):
    test_dataset = aedataset(datafile=datafile, modality="T1", transforms=transforms)
    rand_index = random.randint(0, len(test_dataset) - 1)
    print(f"Using sample index: {rand_index}")
    sample = test_dataset[rand_index]
    x, mask = sample
    x = x.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    original = x * mask

    reconstructions = []
    metrics = []

    # PyTorch models
    for path in pt_paths:
        model = engine_AE.load_from_checkpoint(path).to(device)
        model.eval()
        with torch.no_grad():
            recon, _ = model(x)
            recon = recon * mask
        recon_cpu = recon.squeeze().cpu()
        reconstructions.append(recon_cpu.numpy())
        metrics.append(compute_metrics(original, recon, mask))

    # Keras model
    keras_model = tf.keras.models.load_model(keras_path)
    x_np = x.squeeze().cpu().numpy()[None, ..., None]  # shape: (1, D, H, W, 1)
    recon_np = keras_model.predict(x_np)[0, ..., 0]    # remove batch and channel dims
    mask_np = mask.squeeze().cpu().numpy()
    recon_np_masked = recon_np * mask_np
    reconstructions.append(recon_np_masked)
    keras_metrics = compute_metrics(original, torch.tensor(recon_np_masked).unsqueeze(0), mask)
    metrics.append(keras_metrics)

    # Print metrics
    model_names = pt_names + [keras_name]
    for name, (mse, mae, psnr_val, ssim_val) in zip(model_names, metrics):
        print(f"Model: {name}")
        print(f"  MSE:   {mse:.6f}")
        print(f"  MAE:   {mae:.6f}")
        print(f"  PSNR:  {psnr_val:.2f}")
        print(f"  SSIM:  {ssim_val:.4f}\n")

    # Plotting
    plot_comparisons(
        original.squeeze().cpu().numpy(),
        reconstructions,
        model_names,
        title="Model Comparison on Random Test Sample",
    )

if __name__ == "__main__":
    transforms_monai = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        Resize((182, 218, 182)),  # Adapt shape as needed
    ])

    pt_paths = [
        "checkpoints/model1.ckpt",
        "checkpoints/model2.ckpt"
    ]
    pt_names = ["Model 1 (PT)", "Model 2 (PT)"]
    keras_path = "checkpoints/model3.keras"
    keras_name = "Model 3 (Keras)"
    test_datafile = "iopaths/test_data.txt"

    run_comparison(pt_paths, pt_names, keras_path, keras_name, test_datafile, transforms_monai)






