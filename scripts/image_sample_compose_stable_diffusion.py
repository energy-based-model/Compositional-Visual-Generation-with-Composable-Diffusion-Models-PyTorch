import torch as th
import numpy as np
import torchvision.utils as tvu

from composable_diffusion.composable_stable_diffusion.pipeline_composable_stable_diffusion import \
    ComposableStableDiffusionPipeline

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--prompts", type=str, nargs="+", default=["mystical trees", "A magical pond", "dark"])
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--scale", type=float, default=7.5)
parser.add_argument('--weights', type=float, nargs="+", default=7.5)
parser.add_argument("--seed", type=int, default=8)
parser.add_argument("--model_path", type=str, default="CompVis/stable-diffusion-v1-4")
parser.add_argument("--num_images", type=int, default=1)
args = parser.parse_args()

has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')

prompts = args.prompts
weights = args.weights
scale = args.scale
steps = args.steps

assert len(weights) == 1 or len(weights) == len(prompts), \
    "the number of weights should be the same as the number of prompts."

pipe = ComposableStableDiffusionPipeline.from_pretrained(
    args.model_path,
).to(device)

pipe.safety_checker = None

images = []
generator = th.Generator("cuda").manual_seed(args.seed)
for i in range(args.num_images):
    image = pipe(prompt, guidance_scale=scale, num_inference_steps=steps,
                 weights=args.weights, generator=generator).images[0]
    images.append(th.from_numpy(np.array(image)).permute(2, 0, 1) / 255.)
grid = tvu.make_grid(th.stack(images, dim=0), nrow=4, padding=0)
tvu.save_image(grid, f'{prompt}_{args.weights}' + '.png')
