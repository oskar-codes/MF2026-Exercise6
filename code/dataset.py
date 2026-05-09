from torchvision.transforms import InterpolationMode
import torch
import torchvision
import os

class SRDataset(torch.utils.data.Dataset):
  def __init__(self, path: str, augment: bool = True) -> None:
    super().__init__()

    self.augment = augment

    # Create a list of filenames in the directory
    self.filenames = [f"{path}/{filename}" for filename in os.listdir(path)]

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    # Load the data from the file
    image_path = self.filenames[idx]
    image = torchvision.io.read_image(image_path)

    # Convert the image to a tensor and normalize it
    image = image.float() / 255.0

    if self.augment:
      # Randomly crop a 64x64 square
      crop = torchvision.transforms.RandomCrop((64, 64))

      # ColorJitter
      brightness = 0.2
      contrast = 0.2
      saturation = 0.2
      hue = 0.2
      color_jitter = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)

      transforms = torchvision.transforms.Compose([
        crop,
        color_jitter,
      ])
    else:
      # No augmentation, used for evaluation
      transforms = torchvision.transforms.CenterCrop((64, 64))

    # Bilinearly downscale the image by 2
    resize = torchvision.transforms.Resize(
      (32, 32),
      interpolation=InterpolationMode.BILINEAR,
      antialias=True
    )

    image = transforms(image)
    reference_image = image.clone()

    image = resize(image)


    return image, reference_image


