import torch as th
import numpy as np
import torchvision.utils as tvu

from diffusers import LMSDiscreteScheduler, DDIMScheduler, DDPMScheduler, PNDMScheduler
from composable_diffusion.composable_stable_diffusion.pipeline_composable_stable_diffusion import \
    ComposableStableDiffusionPipeline

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--prompts", type=str, default="mystical trees | A magical pond | dark")
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--scale", type=float, default=7.5)
parser.add_argument('--weights', type=str, default="7.5 | 7.5 | 7.5")
parser.add_argument("--seed", type=int, default=8)
parser.add_argument("--model_path", type=str, default="CompVis/stable-diffusion-v1-4")
parser.add_argument("--num_images", type=int, default=1)
parser.add_argument("--scheduler", type=str, choices=["lms", "ddim", "ddpm", "pndm"], default="ddim",
                    help="ddpm may generate pure noises when using fewer steps.")
args = parser.parse_args()

has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')

prompts = args.prompts
weights = args.weights
scale = args.scale
steps = args.steps

pipe = ComposableStableDiffusionPipeline.from_pretrained(
    args.model_path,
).to(device)

# you can find more schedulers from https://github.com/huggingface/diffusers/blob/main/src/diffusers/__init__.py#L54
if args.scheduler == "lms":
    pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
elif args.scheduler == "ddim":
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
elif args.scheduler == "ddpm":
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
elif args.scheduler == "pndm":
    pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)

pipe.safety_checker = None

images = []
generator = th.Generator("cuda").manual_seed(args.seed)
for i in range(args.num_images):
    image = pipe(prompts, guidance_scale=scale, num_inference_steps=steps,
                 weights=args.weights, generator=generator).images[0]
    images.append(th.from_numpy(np.array(image)).permute(2, 0, 1) / 255.)
grid = tvu.make_grid(th.stack(images, dim=0), nrow=4, padding=0)
tvu.save_image(grid, f'{args.prompts}_{args.weights}' + '.png')
