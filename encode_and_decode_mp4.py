#!/usr/bin/env python3

import argparse
import math
import os
from PIL import Image
import av

import numpy as np
import torch
import torch.distributed.optim
import torch.utils.checkpoint
import torch.utils.data
import torchvision.transforms.v2.functional as transforms_f
from einops import rearrange
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from magvit2.config import VQConfig
from magvit2.models.lfqgan import VQModel


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize tokenized video as GIF or comic.")
    parser.add_argument(
        "--video_path",
        type=str,
        help="Path of the mp4 file",
    )
    parser.add_argument(
        "--ckpt_path", type=str, help="Path to the ckpt file"
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second"
    )
    args = parser.parse_args()

    return args


def open_video(file):
    container = av.open(file)
    video = []

    for frame in container.decode(video=0):
        # Convert frame to numpy array in RGB format
        rgb_image = frame.to_rgb().to_ndarray()
        video.append(rgb_image)

    container.close()
    return torch.from_numpy(np.stack(video))


def export_to_gif(frames: list, output_gif_path: str, fps: int):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - fps (int): Desired frames per second.
    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    duration_ms = 1000 / fps
    pil_frames[0].save(output_gif_path.replace(".mp4", ".gif"),
                       format="GIF",
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=duration_ms,
                       loop=0)


def decode_latents_wrapper(video_data, ckpt_path, batch_size=16):
    """
    video_data: (t, h, w, c)
    """
    device = "cuda"
    dtype = torch.bfloat16

    model_config = VQConfig()
    model = VQModel(model_config, ckpt_path=ckpt_path)
    model = model.to(device=device, dtype=dtype)

    decoded_imgs = []

    video_data = rearrange(video_data, 't h w c -> t c h w').to(device).to(dtype) / 127.5 - 1
    for shard_ind in range(math.ceil(len(video_data) / batch_size)):
        batch = video_data[shard_ind * batch_size: (shard_ind + 1) * batch_size]
        with torch.no_grad():
            quant = model.quantize(model.encode(batch)[0])[0]
            decoded_imgs.append(((model.decode(quant).detach().cpu() + 1) * 127.5).to(dtype=torch.uint8))

    return [transforms_f.to_pil_image(img) for img in torch.cat(decoded_imgs)]


@torch.no_grad()
def main():
    args = parse_args()

    frames = open_video(args.video_path)
    decoded_frames = decode_latents_wrapper(frames, ckpt_path=args.ckpt_path)
    export_to_gif(frames.numpy(), f"original_{args.video_path}", args.fps)
    export_to_gif(decoded_frames, f"decoded_{args.video_path}", args.fps)


if __name__ == "__main__":
    main()
