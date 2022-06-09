from PIL import Image
import torch as th
import argparse

from composable_diffusion.download import load_checkpoint
from composable_diffusion.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)

parser = argparse.ArgumentParser()
parser.add_argument('--timestep_respacing', type=int, default=100)
parser.add_argument('--guidance_scale', type=float, default=10)
parser.add_argument('--upsample_temp', type=float, default=0.98)
parser.add_argument('--prompt', type=str, default='a camel | a forest', help="using `|` to compose multiple sentences")
args = parser.parse_args()

has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')

# Create base model.
options = model_and_diffusion_defaults()
options['use_fp16'] = has_cuda
options['timestep_respacing'] = str(args.timestep_respacing)
model, diffusion = create_model_and_diffusion(**options)
model.eval()
if has_cuda:
    model.convert_to_fp16()
model.to(device)
model.load_state_dict(load_checkpoint('base', device))
print('total base parameters', sum(x.numel() for x in model.parameters()))

# Create upsampler model.
options_up = model_and_diffusion_defaults_upsampler()
options_up['use_fp16'] = has_cuda
options_up['timestep_respacing'] = 'fast27' # use 27 diffusion steps for very fast sampling
model_up, diffusion_up = create_model_and_diffusion(**options_up)
model_up.eval()
if has_cuda:
    model_up.convert_to_fp16()
model_up.to(device)
model_up.load_state_dict(load_checkpoint('upsample', device))
print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))


def show_images(batch: th.Tensor, file_name):
    """ Display a batch of images inline. """
    scaled = ((batch + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    Image.fromarray(reshaped.numpy()).save(file_name)

# Sampling parameters
prompt = args.prompt
prompts = [x.strip() for x in prompt.split('|')]
batch_size = 1
guidance_scale = args.guidance_scale
# Tune this parameter to control the sharpness of 256x256 images.
# A value of 1.0 is sharper, but sometimes results in grainy artifacts.
upsample_temp = args.upsample_temp

##############################
# Sample from the base model #
##############################

# Create the text tokens to feed to the model.
tokens_list = [model.tokenizer.encode(prompt) for prompt in prompts]
outputs = [model.tokenizer.padded_tokens_and_mask(
    tokens, options['text_ctx']
) for tokens in tokens_list]

cond_tokens, cond_masks = zip(*outputs)
cond_tokens, cond_masks = list(cond_tokens), list(cond_masks)

full_batch_size = batch_size * (len(prompts) + 1)
uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
    [], options['text_ctx']
)

# Pack the tokens together into model kwargs.
model_kwargs = dict(
    tokens=th.tensor(
        cond_tokens + [uncond_tokens], device=device
    ),
    mask=th.tensor(
        cond_masks + [uncond_mask],
        dtype=th.bool,
        device=device,
    ),
)

masks = [True] * len(prompts) + [False]
# coefficients = th.tensor([0.5, 0.5], device=device).reshape(-1, 1, 1, 1)
masks = th.tensor(masks, dtype=th.bool, device=device)

# Create a classifier-free guidance sampling function
def model_fn(x_t, ts, **kwargs):
    half = x_t[:1]
    combined = th.cat([half] * x_t.size(0), dim=0)
    model_out = model(combined, ts, **kwargs)
    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps = eps[masks].mean(dim=0, keepdim=True)
    # cond_eps = (coefficients * eps[masks]).sum(dim=0)[None]
    uncond_eps = eps[~masks].mean(dim=0, keepdim=True)
    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
    eps = th.cat([half_eps] * x_t.size(0), dim=0)
    return th.cat([eps, rest], dim=1)

# Sample from the base model.
model.del_cache()
samples = diffusion.p_sample_loop(
    model_fn,
    (full_batch_size, 3, options["image_size"], options["image_size"]),
    device=device,
    clip_denoised=True,
    progress=True,
    model_kwargs=model_kwargs,
    cond_fn=None,
)[:batch_size]
model.del_cache()

# Show the output
show_images(samples, 'result_64.png')

##############################
# Upsample the 64x64 samples #
##############################

tokens = model_up.tokenizer.encode("".join(prompts))
tokens, mask = model_up.tokenizer.padded_tokens_and_mask(
    tokens, options_up['text_ctx']
)

# Create the model conditioning dict.
model_kwargs = dict(
    # Low-res image to upsample.
    low_res=((samples+1)*127.5).round()/127.5 - 1,

    # Text tokens
    tokens=th.tensor(
        [tokens] * batch_size, device=device
    ),
    mask=th.tensor(
        [mask] * batch_size,
        dtype=th.bool,
        device=device,
    ),
)

# Sample from the base model.
model_up.del_cache()
up_shape = (batch_size, 3, options_up["image_size"], options_up["image_size"])
up_samples = diffusion_up.ddim_sample_loop(
    model_up,
    up_shape,
    noise=th.randn(up_shape, device=device) * upsample_temp,
    device=device,
    clip_denoised=True,
    progress=True,
    model_kwargs=model_kwargs,
    cond_fn=None,
)[:batch_size]
model_up.del_cache()

# Show the output
show_images(up_samples, 'result_upsampled_256.png')
