# - Run the evaluation on the provided eval dataset. To set a baseline, do the evaluation with
# traditional bilinear and bicubic upscaling methods. Do not augment the evaluation images.
# - Load a pre-trained model from a file using torch.load and the method .load state dict,
# run the evaluation with this model.
# - Use PSNR and SSIM as metrics (you can use the implementation available in the package
# pytorch-msssim).

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from dataset import SRDataset
from sr_model import BasicSRModel
import argparse

def evaluate(model, dataloader, device):
    model.eval()
    psnr_total = 0
    ssim_total = 0
    with torch.no_grad():
        for lr, hr in dataloader:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            psnr_total += 10 * torch.log10(1 / F.mse_loss(sr, hr))
            ssim_total += ssim(sr, hr, data_range=1.0)
    avg_psnr = psnr_total / len(dataloader)
    avg_ssim = ssim_total / len(dataloader)
    return avg_psnr.item(), avg_ssim.item()

def main(checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = "data/eval"
    eval_dataset = SRDataset(data_path)

    eval_loader = torch.utils.data.DataLoader(
      eval_dataset,
      batch_size=4,
      shuffle=True,
      num_workers=2,
      drop_last=True,
      pin_memory=True,
    )

    # Load pre-trained model
    model = BasicSRModel()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    # Evaluate the model
    psnr, ssim_value = evaluate(model, eval_loader, device)
    print(f'PSNR: {psnr:.2f} dB, SSIM: {ssim_value:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a super-resolution model')
    parser.add_argument('--run', type=str, required=True, help='Name of the run to evaluate (e.g., "run_2024-06-01_12-00-00")')
    parser.add_argument('--checkpoint', type=str, default="model.pth", help='Specific checkpoint in the run')
    args = parser.parse_args()

    checkpoint_path = f"checkpoints/{args.run}/{args.checkpoint}"
    print(f"Evaluating model at: {checkpoint_path}")

    main(checkpoint_path)
