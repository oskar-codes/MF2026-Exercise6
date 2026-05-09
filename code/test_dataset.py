import torch
import torchvision

from dataset import SRDataset

if __name__ == '__main__':
  data_path = "data/train"
  train_dataset = SRDataset(data_path)

  train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2,
    drop_last=True,
    pin_memory=True,
  )
  print(f" * Dataset contains {len(train_dataset)} image(s).")

  for _, batch in enumerate(train_dataloader, 0):
    lr_image, hr_image = batch
    torchvision.io.write_png(lr_image[0, ...].mul(255).byte(), "lr_image.png")
    torchvision.io.write_png(hr_image[0, ...].mul(255).byte(), "hr_image.png")
    break # we deliberately break after one batch as this is just a test
