import os

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from dataset import SRDataset
from sr_model import BasicSRModel
from res_model import ResSRModel
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
import argparse

number_of_epochs = 200
learning_rate = 1e-4

if __name__ == '__main__':
  start_time = datetime.now()

  parser = argparse.ArgumentParser(description='Train the super-resolution model')
  parser.add_argument('--epochs', type=int, default=number_of_epochs, help='Number of training epochs')
  parser.add_argument('--lr', type=float, default=learning_rate, help='Learning rate for the optimizer')
  parser.add_argument('--model', type=str, default='basic', choices=['basic', 'residual'], help='Model architecture to use (basic or residual)')
  args = parser.parse_args()
  number_of_epochs = args.epochs
  learning_rate = args.lr
  model_choice = args.model

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print('Device: {}'.format(device))
  if device == 'cuda':
    print(f"- GPU: {torch.cuda.get_device_name(0)}")
  print(f"[model] {model_choice} | [epochs] {number_of_epochs} | [learning_rate] {learning_rate}")

  current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  writer = SummaryWriter(f"runs/experiment_{current_time}")

  train_dataset = SRDataset("data/train")
  val_dataset = SRDataset("data/eval")

  train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2,
    drop_last=True,
    pin_memory=True,
  )

  val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=2,
    drop_last=True,
    pin_memory=True,
  )

  model = BasicSRModel() if model_choice == 'basic' else ResSRModel()
  model.to(device)

  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

  loss_function = torch.nn.L1Loss().to(device)

  global_step = 0

  for epoch in range(number_of_epochs):
    print(f"=== Epoch {epoch+1}/{number_of_epochs} ===")

    # --- Training ---
    model.train()
    losses = []
    for _, batch in enumerate(train_dataloader):
      low_res, high_res = batch[0].to(device), batch[1].to(device)
      optimizer.zero_grad()
      high_res_prediction = model(low_res)
      loss = loss_function(high_res_prediction, high_res)
      loss.backward()
      optimizer.step()
      losses.append(loss.item())
      writer.add_scalar("Loss/train", loss.item(), global_step)
      global_step += 1

    avg_train_loss = sum(losses) / len(losses)
    print(f"  -> Train L1: {avg_train_loss:.4f}")

    # --- Validation ---
    val_l1_total = 0
    psnr_total = 0
    ssim_total = 0
    model.eval()
    with torch.no_grad():
      for lr, hr in val_dataloader:
        lr, hr = lr.to(device), hr.to(device)
        sr = model(lr)
        val_l1_total += loss_function(sr, hr).item()
        psnr_total += 10 * torch.log10(1 / F.mse_loss(sr, hr)).item()
        ssim_total += ssim(sr, hr, data_range=1.0).item()

    n = len(val_dataloader)
    avg_val_l1 = val_l1_total / n
    avg_val_psnr = psnr_total / n
    avg_val_ssim = ssim_total / n
    print(f"  -> Val L1: {avg_val_l1:.4f} | PSNR: {avg_val_psnr:.2f} dB | SSIM: {avg_val_ssim:.4f}")

    writer.add_scalar("Loss/val", avg_val_l1, global_step)
    writer.add_scalar("Metrics/PSNR", avg_val_psnr, global_step)
    writer.add_scalar("Metrics/SSIM", avg_val_ssim, global_step)

    # Periodically save the model checkpoint
    if (epoch + 1) % 50 == 0:
      print(f"  -> Saving checkpoint for epoch {epoch + 1}...")
      filename = f"checkpoints/{current_time}/checkpoint_{epoch + 1}.pth"
      os.makedirs(os.path.dirname(filename), exist_ok=True)
      torch.save(model.state_dict(), filename)

  final_filename = f"checkpoints/{current_time}/model.pth"
  torch.save(model.state_dict(), final_filename)

  writer.close()

  end_time = datetime.now()
  minutes = (end_time - start_time).total_seconds() / 60
  seconds = (end_time - start_time).total_seconds() % 60
  print(f"Training completed in {int(minutes)}:{int(seconds):02d}")
