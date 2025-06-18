# Copyright (c) 2025, Weijian Luo. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
# https://www.apache.org/licenses/LICENSE-2.0.txt

import os
import re
import gc
import copy
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import distributed as dist
from inference_helper_func import load_sd3, sd3_sampler
from diffusers import StableDiffusion3Pipeline


class StackedRandomGenerator:
    """Wrapper for torch.Generator supporting different seeds per minibatch sample"""
    def __init__(self, device, seeds):
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32))
            for seed in seeds
        ]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([
            torch.randn(size[1:], generator=gen, **kwargs)
            for gen in self.generators
        ])

    def randn_like(self, input):
        return self.randn(
            input.shape, 
            dtype=input.dtype,
            layout=input.layout,
            device=input.device
        )


def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

def read_prompts(filename):
    """Read text prompts from file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def save_as_npz(folder_path, max_images=50000):
    """Compress folder images to NPZ file"""
    image_exts = ('.png', '.jpg', '.jpeg')
    image_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(image_exts)
    ][:max_images]

    images = []
    for filename in tqdm.tqdm(image_files, desc="Compressing images"):
        img = PIL.Image.open(os.path.join(folder_path, filename))
        images.append(np.array(img))

    npz_path = f"{folder_path}.npz"
    np.savez(npz_path, arr_0=np.stack(images))
    print(f"Saved {len(images)} images to {npz_path}")


@click.command()
@click.option('--network', 'network_pkl', required=True,
              help='Network pickle filename', metavar='PATH|URL')
@click.option('--outdir', required=True,
              help='Output directory', metavar='DIR')
@click.option('--seeds', default='0-63', show_default=True,
              help='Random seeds', metavar='LIST', type=parse_int_list)
@click.option('--subdirs', is_flag=True,
              help='Create subdirectories for every 1000 seeds')
@click.option('--batch', 'max_batch_size', default=16, show_default=True,
              help='Maximum batch size', type=click.IntRange(min=1))
@click.option('--init_timestep', default=0, show_default=True,
              help='Initial timestep (0-999)', type=click.IntRange(min=0))
@click.option('--resolution', default=1024, show_default=True,
              help='generation resolution', type=click.IntRange(min=0))
@click.option('--out_resolution', default=1024, show_default=True,
              help='save resolution', type=click.IntRange(min=0))
@click.option('--text_prompts', default='prompts/captions.txt', show_default=True,
              help='Prompt file', metavar='PATH')
@click.option('--repo_id', default='runwayml/stable-diffusion-v1-5', show_default=True,
              help='Diffusion pipeline ID', metavar='ID')
@click.option('--use_fp16', default=True, show_default=True,
              help='Enable mixed-precision', type=bool)
@click.option('--enable_xformers', default=True, show_default=True,
              help='Enable xformers', type=bool)
@click.option('--custom_seed', is_flag=True,
              help='Use custom seeds')
@click.option('--compile', is_flag=True, help='Enable JIT compilation with torch.compile')
def main(
    network_pkl,
    outdir,
    seeds,
    subdirs,
    max_batch_size,
    resolution,
    out_resolution,
    init_timestep,
    text_prompts,
    repo_id,
    compile,
    use_fp16=True,
    enable_xformers=True, 
    custom_seed=False,
):
    """Generate images from one-step models"""
    dist.init()
    device = torch.device('cuda')
    dtype = torch.bfloat16 if use_fp16 else torch.float32
    captions = read_prompts(text_prompts)

    # Prepare seed batches
    seeds = parse_int_list(args.seeds)[0:len(captions)]
    num_batches = len(seeds)//args.batch
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank()::dist.get_world_size()]

    torch.distributed.barrier()

    pipe = StableDiffusion3Pipeline.from_pretrained(repo_id, torch_dtype=torch.bfloat16)
    pipe_kwargs = dict(guidance_scale=4.5, num_inference_steps=40, )
    pipe = pipe.to(device)


    # Create output directory
    output_dir = outdir
    dist.print0(f'Generating {len(seeds)} images to "{output_dir}"...')

    torch.distributed.barrier()

    # Generate images
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Prepare seeds and captions
        if custom_seed:
            actual_seeds = [seeds[i] for i in batch_seeds]
        else:
            actual_seeds = batch_seeds.tolist()
            
        rnd = StackedRandomGenerator(device, actual_seeds)
        batch_captions = [captions[i] for i in actual_seeds]

        with torch.no_grad():
            images = pipe(prompt=batch_captions, **pipe_kwargs).images

        for seed, image, caption in zip(actual_seeds, images, batch_captions):
            subdir = os.path.join(output_dir, f'{seed//1000*1000:06d}') if subdirs else output_dir
            os.makedirs(subdir, exist_ok=True)
            
            base_path = os.path.join(subdir, f'{seed:06d}')

            image.resize((out_resolution, out_resolution)).save(f'{base_path}.png')
            
            with open(f'{base_path}.txt', 'w', encoding='utf-8') as f:  # ADDED ENCODING
                f.write(caption)

    torch.distributed.barrier()
    dist.print0('Done.')
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
    # torchrun --standalone --nproc_per_node=2 \
    # generate_sd3_baseline.py --init_timestep 120 \
    # --text_prompts 'prompts/parti_prompts.txt' \
    # --repo_id 'stabilityai/stable-diffusion-3.5-medium' \
    # --outdir='evaluate/parti_sd3/sd3_cfg4.5_steps40_512px' \
    # --seeds=0-1631 --batch=8 --network='None' --enable_xformers=0 \
    # --use_fp16=1 --out_resolution 512
