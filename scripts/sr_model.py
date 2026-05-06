
import torch
import torch.nn as nn

class BasicSRModel(torch.nn.Module):
  def __init__(self, blocks = 10, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)


    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    self.first = nn.Conv2d(
      in_channels=3,
      out_channels=64,
      kernel_size=3,
      padding=1,
    )

    # Blocks of Conv2d -> LeakyReLU (use nn.Sequential for this)
    self.blocks = nn.Sequential(*[
      nn.Sequential(
        nn.Conv2d(
          in_channels=64,
          out_channels=64,
          kernel_size=3,
          padding=1,
        ),
        nn.LeakyReLU(),
      )
      for _ in range(blocks)
    ])

    self.last = nn.Conv2d(
      in_channels=64,
      out_channels=3,
      kernel_size=3,
      padding=1,
    )

  def forward(self, x):
    x = self.upsample(x)
    x = self.first(x)
    x = self.blocks(x)
    x = self.last(x)
    return x
  

if __name__ == '__main__':
  model = BasicSRModel(blocks = 10)
  num_params = 0
  for param in model.parameters():
    num_params += param.numel()
  print(num_params)

