import os

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from dataset import SRDataset
from sr_model import BasicSRModel
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime

number_of_epochs = 50
learning_rate = 0.001

if __name__ == '__main__':
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print('Using {} device'.format(device))
  if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

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

  model = BasicSRModel()
  model.to(device)

  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

  loss_function = torch.nn.L1Loss().to(device)

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

    writer.add_scalar("Loss/train", avg_train_loss, epoch)
    writer.add_scalar("Loss/val", avg_val_l1, epoch)
    writer.add_scalar("Metrics/PSNR", avg_val_psnr, epoch)
    writer.add_scalar("Metrics/SSIM", avg_val_ssim, epoch)

    # Periodically save the model checkpoint
    if (epoch + 1) % 2 == 0 or (epoch + 1) == number_of_epochs:
      print(f"  -> Saving checkpoint for epoch {epoch + 1}...")
      filename = f"checkpoints/{current_time}/checkpoint_{epoch + 1}.pth"
      os.makedirs(os.path.dirname(filename), exist_ok=True)
      torch.save(model.state_dict(), filename)

  writer.close()
