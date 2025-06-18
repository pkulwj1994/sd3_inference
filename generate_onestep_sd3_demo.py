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
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank()::dist.get_world_size()]

    # wait for everyone
    torch.distributed.barrier()
    
    ## load SD3.5 model
    G_ema, vae, noise_scheduler, text_encoder_one, text_encoder_two, text_encoder_three, tokenizer_one, tokenizer_two, tokenizer_three, image_processor = load_sd3(pretrained_model_name_or_path=repo_id, pretrained_vae_model_name_or_path=repo_id,
                                                            device=device, weight_dtype=dtype, enable_xformers=enable_xformers,)
    
    sample_scheduler = copy.deepcopy(noise_scheduler)
    gen_timestep = noise_scheduler.timesteps[init_timestep]

    latent_img_channels = G_ema.config.in_channels
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    latent_resolution = resolution // vae_scale_factor

    # Load models
    if dist.get_rank() == 0:
        dist.print0(f'Loading network from "{network_pkl}"...')

    G_ema.load_state_dict(torch.load(network_pkl))
    G_ema.eval().requires_grad_(False)

    gc.collect(); torch.cuda.empty_cache()

    # Create output directory
    output_dir = outdir
    dist.print0(f'Generating {len(seeds)} images to "{output_dir}"...')

    # wait for every rank
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

        # Generate latent vectors
        latents = rnd.randn([len(actual_seeds), latent_img_channels, latent_resolution, latent_resolution], device=device, dtype=dtype)
        gen_timesteps = gen_timestep * torch.ones((len(batch_captions),), device=device, dtype=torch.float32)

        # Run generator
        with torch.inference_mode():
            images = sd3_sampler(
                transformer=G_ema,
                latents=latents,
                contexts=batch_captions,
                gen_timesteps=gen_timesteps,
                noise_scheduler=sample_scheduler,
                text_encoder_one=text_encoder_one, 
                text_encoder_two=text_encoder_two, 
                text_encoder_three=text_encoder_three, 
                tokenizer_one=tokenizer_one, 
                tokenizer_two=tokenizer_two, 
                tokenizer_three=tokenizer_three, 
                image_processor=image_processor,
                resolution=None, ## not used
                dtype=dtype, 
                return_images=False, 
                vae=None
                )

        images = (images / vae.config.scaling_factor) + vae.config.shift_factor
        images = vae.decode(images, return_dict=False)[0]
        images = image_processor.postprocess(images, output_type='pt')

        images = (images * 255).clamp(0, 255).to(torch.uint8)
        images_np = images.permute(0, 2, 3, 1).cpu().numpy()

        for seed, image_np, caption in zip(actual_seeds, images_np, batch_captions):
            subdir = os.path.join(output_dir, f'{seed//1000*1000:06d}') if subdirs else output_dir
            os.makedirs(subdir, exist_ok=True)
            
            base_path = os.path.join(subdir, f'{seed:06d}')

            PIL.Image.fromarray(image_np).resize((out_resolution, out_resolution)).save(f'{base_path}.png')
            
            with open(f'{base_path}.txt', 'w', encoding='utf-8') as f:  # ADDED ENCODING
                f.write(caption)


    dist.print0('Done.')
    # wait for everyone before close
    torch.distributed.barrier()


if __name__ == "__main__":
    main()
