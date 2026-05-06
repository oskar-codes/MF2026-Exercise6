import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
from dataset import SRDataset
from sr_model import BasicSRModel

number_of_epochs = 10
learning_rate = 0.001

if __name__ == '__main__':
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print('Using {} device'.format(device))
  if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

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

  train_losses = []
  val_losses = []
  val_psnrs = []
  val_ssims = []

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
    train_losses.append(avg_train_loss)
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
    val_losses.append(val_l1_total / n)
    val_psnrs.append(psnr_total / n)
    val_ssims.append(ssim_total / n)
    print(f"  -> Val L1: {val_losses[-1]:.4f} | PSNR: {val_psnrs[-1]:.2f} dB | SSIM: {val_ssims[-1]:.4f}")

    # Periodically save the model checkpoint
    if (epoch + 1) % 5 == 0 or (epoch + 1) == number_of_epochs:
      print(f"  -> Saving checkpoint for epoch {epoch + 1}...")
      torch.save(model.state_dict(), f"checkpoints/checkpoint_epoch_{epoch + 1}.pth")

  epochs = range(1, number_of_epochs + 1)
  fig, axes = plt.subplots(2, 2, figsize=(12, 8))

  axes[0, 0].plot(epochs, train_losses)
  axes[0, 0].set_title("Training L1 Loss")
  axes[0, 0].set_xlabel("Epoch")
  axes[0, 0].set_ylabel("L1 Loss")

  axes[0, 1].plot(epochs, val_losses)
  axes[0, 1].set_title("Validation L1 Loss")
  axes[0, 1].set_xlabel("Epoch")
  axes[0, 1].set_ylabel("L1 Loss")

  axes[1, 0].plot(epochs, val_psnrs)
  axes[1, 0].set_title("Validation PSNR")
  axes[1, 0].set_xlabel("Epoch")
  axes[1, 0].set_ylabel("PSNR (dB)")

  axes[1, 1].plot(epochs, val_ssims)
  axes[1, 1].set_title("Validation SSIM")
  axes[1, 1].set_xlabel("Epoch")
  axes[1, 1].set_ylabel("SSIM")

  plt.tight_layout()
  plt.show()
