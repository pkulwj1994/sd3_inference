import torch
import diffusers
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
)
from packaging import version
from diffusers.utils.import_utils import is_xformers_available


# from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
def upcast_vae(vae):
    dtype = vae.dtype
    vae.to(dtype=torch.float32)

@torch.no_grad()
def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def tokenize_captions(tokenizer, examples):
    max_length = tokenizer.model_max_length
    captions = []
    for caption in examples:
        captions.append(caption)

    text_inputs = tokenizer(
        captions, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt"
    )

    return text_inputs.input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    text_input_ids=None,
    num_images_per_prompt=1,
    device=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    if prompt is not None:
        batch_size = len(prompt)
    else:
        batch_size = text_input_ids.shape[0]

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds

def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    if prompt is not None:
        batch_size = len(prompt)
    else:
        batch_size = text_input_ids.shape[0]

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds

### FIXME need double check
# Adapted from pipelines.StableDiffusion3Pipeline.encode_prompt
@torch.no_grad()
def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length=77,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    if tokenizers is not None:
        clip_tokenizers = tokenizers[:2]
        t5_tokenizer = tokenizers[-1]
    else:
        clip_tokenizers = [None, None]
        t5_tokenizer = None

    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for i, (tokenizer, text_encoder) in enumerate(zip(clip_tokenizers, clip_text_encoders)):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=text_input_ids_list[i] if text_input_ids_list else None,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoder=text_encoders[-1],
        tokenizer=t5_tokenizer,
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[-1] if text_input_ids_list else None,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds



def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection

    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def get_sigmas(noise_scheduler, timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=timesteps.device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device=timesteps.device)
    # print(f"schedule_timesteps: {schedule_timesteps}")
    timesteps = timesteps.to(device=timesteps.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

def load_sd3(pretrained_model_name_or_path, pretrained_vae_model_name_or_path, device, weight_dtype, 
              revision=None, variant=None, lora_config=None, enable_xformers=False, gradient_checkpointing=False):
    # Load the tokenizer
    print(f'pretrained_model_name_or_path: {pretrained_model_name_or_path}')
    print(f'pretrained_vae_name_or_path: {pretrained_vae_model_name_or_path}')
    print(f'revision: {revision}')

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        pretrained_vae_model_name_or_path,
        subfolder="tokenizer",
        revision=revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        pretrained_vae_model_name_or_path,
        subfolder="tokenizer_2",
        revision=revision,
        use_fast=False,
    )
    tokenizer_three = AutoTokenizer.from_pretrained(
        pretrained_vae_model_name_or_path,
        subfolder="tokenizer_3",
        revision=revision,
        use_fast=False,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        pretrained_vae_model_name_or_path, revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        pretrained_vae_model_name_or_path, revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        pretrained_vae_model_name_or_path, revision, subfolder="text_encoder_3"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained_vae_model_name_or_path, subfolder="scheduler")
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        pretrained_vae_model_name_or_path, subfolder="text_encoder", revision=revision, variant=variant
    )

    text_encoder_two = text_encoder_cls_two.from_pretrained(
        pretrained_vae_model_name_or_path, subfolder="text_encoder_2", revision=revision, variant=variant
    )

    text_encoder_three = text_encoder_cls_three.from_pretrained(
        pretrained_vae_model_name_or_path, subfolder="text_encoder_3", revision=revision, variant=variant
    )

    vae_path = (
        pretrained_vae_model_name_or_path
        if pretrained_vae_model_name_or_path is not None
        else pretrained_model_name_or_path
    )

    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae",
        revision=revision,
        variant=variant,
    )

    transformer = SD3Transformer2DModel.from_pretrained(pretrained_model_name_or_path, subfolder="transformer", revision=revision, variant=variant)

    # Freeze untrained components
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)

    # Move transformer and text_encoders to device and cast to weight_dtype
    transformer.to(device, dtype=weight_dtype)
    text_encoder_one.to(device, dtype=weight_dtype)
    text_encoder_two.to(device, dtype=weight_dtype)
    text_encoder_three.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)

    from diffusers.image_processor import PipelineImageInput, VaeImageProcessor # adapted from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py#L219
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    if enable_xformers:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                ValueError(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            transformer.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    return transformer, vae, noise_scheduler, text_encoder_one, text_encoder_two, text_encoder_three, tokenizer_one, tokenizer_two, tokenizer_three, image_processor


def sd3_sampler(transformer, latents, contexts, gen_timesteps,  noise_scheduler, 
                         text_encoder_one, text_encoder_two, text_encoder_three, tokenizer_one, 
                         tokenizer_two, tokenizer_three, image_processor, resolution, dtype=torch.bfloat16,
                         return_images=False, vae=None, guidance_scale=1):
    
    # Get the text embedding for conditioning
    prompt=contexts
    batch_size = len(prompt)

    input_ids_one = tokenize_prompt(tokenizer_one, prompt)
    input_ids_two = tokenize_prompt(tokenizer_two, prompt)    
    input_ids_three = tokenize_prompt(tokenizer_three, prompt)

    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders=[text_encoder_one, text_encoder_two, text_encoder_three],
            tokenizers=None,
            prompt=None,
            text_input_ids_list=[input_ids_one, input_ids_two, input_ids_three],
        )

    # from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
    assert noise_scheduler.config.get("use_dynamic_shifting", None) == False # refer: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py#L1009

    x0 = torch.zeros_like(latents).to(latents.device)

    latents = noise_scheduler.scale_noise(sample=x0, timestep=gen_timesteps, noise=latents) # mix 0 and noise to get a fake xt
    noise_pred = transformer(hidden_states=latents, timestep=gen_timesteps, encoder_hidden_states=prompt_embeds, pooled_projections=pooled_prompt_embeds, return_dict=False)[0].to(dtype)
    sigmas = get_sigmas(noise_scheduler, gen_timesteps, n_dim=latents.ndim, dtype=latents.dtype)

    D_x = latents - sigmas* noise_pred

    return D_x.to(dtype)

def resample(images, t_renoises, transformer, noise_scheduler, prompt_embeds, pooled_prompt_embeds, dtype):
    t_renoises = t_renoises + [999]
    reno_noise = torch.randn_like(images)

    print(t_renoises)

    for i, item in enumerate(zip(t_renoises[:-1],t_renoises[1:])):
        t, t_prev = item
        t_idx = t * torch.ones(size=(images.shape[0],), dtype=torch.float32, device='cpu').long()
        t_timesteps = noise_scheduler.timesteps[t_idx].to(device=images.device)
        t_sigmas = get_sigmas(noise_scheduler, t_timesteps, n_dim=images.ndim, dtype=images.dtype)

        t_idx_prev = t_prev * torch.ones(size=(images.shape[0],), dtype=torch.float32, device='cpu').long()
        t_timesteps_prev = noise_scheduler.timesteps[t_idx_prev].to(device=images.device)
        t_sigmas_prev = get_sigmas(noise_scheduler, t_timesteps_prev, n_dim=images.ndim, dtype=images.dtype)

        if i ==0:
            images, _ = sd3_add_noise(images=images, noise=reno_noise, timesteps=t_timesteps, noise_scheduler=noise_scheduler)
        else:
            pass

        noise_pred = transformer(hidden_states=images, timestep=t_timesteps, encoder_hidden_states=prompt_embeds, pooled_projections=pooled_prompt_embeds, return_dict=False)[0].to(images.dtype)
        images = images - (t_sigmas-t_sigmas_prev)* noise_pred

    return images
