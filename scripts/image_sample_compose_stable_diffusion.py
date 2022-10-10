import torch as th
import numpy as np
import torchvision.utils as tvu

from composable_diffusion.composable_stable_diffusion.pipeline_composable_stable_diffusion import \
    ComposableStableDiffusionPipeline

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="a forest | a camel",
                    help="use '|' as the delimiter to compose separate sentences.")
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--scale", type=float, default=7.5)
parser.add_argument("--weights", type=str, default="")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_images", type=int, default=4)
args = parser.parse_args()

has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')

prompt = args.prompt
scale = args.scale
steps = args.steps

pipe = ComposableStableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=True
).to(device)


def dummy(images, **kwargs):
    return images, False


pipe.safety_checker = dummy

# set prompt weight for each
num_prompts = len(prompt.split("|"))
weights = [float(w.strip()) for w in args.weights.split("|")]
if len(weights) < num_prompts:
    weights.append(1.)
weights = th.tensor(weights, device=device)

images = []
generator = th.Generator("cuda").manual_seed(args.seed)
for i in range(args.num_images):
    image = pipe(prompt, guidance_scale=scale, num_inference_steps=steps, weights=weights, generator=generator)["sample"][0]
    images.append(th.from_numpy(np.array(image)).permute(2, 0, 1) / 255.)
grid = tvu.make_grid(th.stack(images, dim=0), nrow=4, padding=0)
tvu.save_image(grid, f'{prompt}_{args.weights}' + '.png')
