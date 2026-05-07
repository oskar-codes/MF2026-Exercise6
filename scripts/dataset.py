from torchvision.transforms import InterpolationMode
import torch
import torchvision
import os

class SRDataset(torch.utils.data.Dataset):
  def __init__(self, path: str) -> None:
    super().__init__()

    # Create a list of filenames in the directory
    self.filenames = [f"{path}/{filename}" for filename in os.listdir(path)]

  def __len__(self):
    return len(self.filenames)
  
  def __getitem__(self, idx):
    # Load the data from the file
    # Use torchvision.io.read_image(image_path)
    image_path = self.filenames[idx]
    image = torchvision.io.read_image(image_path)

    # Convert the image to a tensor and normalize it
    image = image.float() / 255.0

    # Use torchvision.transforms.RandomCrop, to randomly crop a 64x64 square
    random_crop = torchvision.transforms.RandomCrop((64, 64))

    # Now use torchvision.transforms.ColorJitter
    brightness = 0.2
    contrast = 0.2
    saturation = 0.2
    hue = 0.0
    color_jitter = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)

    # Use torchvision.transforms.Resize to bilinearly downscale the image by 2
    resize = torchvision.transforms.Resize(
      (32, 32),
      interpolation=InterpolationMode.BILINEAR,
      antialias=True
    )

    transforms = torchvision.transforms.Compose([
      random_crop,
      color_jitter,
    ])
    image = transforms(image)
    reference_image = image.clone()
    
    image = resize(image)


    return image, reference_image


