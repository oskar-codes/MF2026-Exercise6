import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from dataset import SRDataset
from sr_model import BasicSRModel
from res_model import ResSRModel
import argparse

def compute_metrics(sr, hr):
    # Computer PSNR and SSIM
    psnr = 10 * torch.log10(1 / F.mse_loss(sr, hr))
    ssim_value = ssim(sr, hr, data_range=1.0)
    return psnr, ssim_value

def evaluate(predict_fn, dataloader, device):
    # Compute average PSNR and SSIM over the dataset
    psnr_total = 0
    ssim_total = 0
    with torch.no_grad():
        for lr, hr in dataloader:
            lr, hr = lr.to(device), hr.to(device)
            sr = predict_fn(lr)
            psnr, ssim_value = compute_metrics(sr, hr)
            psnr_total += psnr
            ssim_total += ssim_value
    avg_psnr = psnr_total / len(dataloader)
    avg_ssim = ssim_total / len(dataloader)
    return avg_psnr.item(), avg_ssim.item()

def upscale(mode):
    # HOF function to create an upscaling function for the given mode
    def _upscale(lr):
        return F.interpolate(lr, scale_factor=2, mode=mode, align_corners=False)
    return _upscale

def main(checkpoint_path, model_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = "data/eval"
    eval_dataset = SRDataset(data_path, augment=False)

    eval_loader = torch.utils.data.DataLoader(
      eval_dataset,
      batch_size=4,
      shuffle=False,
      num_workers=2,
      drop_last=True,
      pin_memory=True,
    )

    # Baselines: traditional bilinear and bicubic upscaling
    for mode in ('bilinear', 'bicubic'):
        psnr, ssim_value = evaluate(upscale(mode), eval_loader, device)
        print(f'[{mode:>8}] PSNR: {psnr:.2f} dB, SSIM: {ssim_value:.4f}')

    # Load pre-trained model
    model = BasicSRModel() if model_type == "basic" else ResSRModel()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Evaluate the model
    psnr, ssim_value = evaluate(model, eval_loader, device)
    print(f'[{"model":>8}] PSNR: {psnr:.2f} dB, SSIM: {ssim_value:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a super-resolution model')
    parser.add_argument('--run', type=str, required=True, help='Name of the run to evaluate (e.g., "run_2024-06-01_12-00-00")')
    parser.add_argument('--checkpoint', type=str, default="model.pth", help='Specific checkpoint in the run')
    parser.add_argument('--model', type=str, default="basic", choices=["basic", "residual"], help='Model type to evaluate')
    args = parser.parse_args()

    checkpoint_path = f"checkpoints/{args.run}/{args.checkpoint}"
    print(f"Evaluating model at: {checkpoint_path}")

    main(checkpoint_path, args.model)
