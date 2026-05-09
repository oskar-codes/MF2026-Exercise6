import os
import argparse

import torch
import torchvision
from sr_model import BasicSRModel


def main(checkpoint_path, image_path, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BasicSRModel()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    image = torchvision.io.read_image(image_path)
    if image.shape[0] == 4:
        image = image[:3]
    image = image.float() / 255.0
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        sr = model(image)

    sr = sr.squeeze(0).clamp(0.0, 1.0).cpu()
    sr = (sr * 255.0).round().to(torch.uint8)
    torchvision.io.write_png(sr, output_path)
    print(f"Saved upscaled image to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run super-resolution inference on an image')
    parser.add_argument('--run', type=str, required=True, help='Name of the run (e.g., "2024-06-01_12-00-00")')
    parser.add_argument('--checkpoint', type=str, default='model.pth', help='Specific checkpoint in the run')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image to upscale')
    parser.add_argument('--output', type=str, default=None, help='Path for the upscaled output image (default: <image>_sr.png)')
    args = parser.parse_args()

    checkpoint_path = f"checkpoints/{args.run}/{args.checkpoint}"

    if args.output is None:
        base, _ = os.path.splitext(args.image)
        output_path = f"{base}_sr.png"
    else:
        output_path = args.output

    print(f"Loading model from: {checkpoint_path}")
    print(f"Upscaling image: {args.image}")

    main(checkpoint_path, args.image, output_path)
