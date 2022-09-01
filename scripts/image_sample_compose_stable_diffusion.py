import torch as th

from torch import autocast
from composable_diffusion.pipeline_composable_stable_diffusion import ComposableStableDiffusionPipeline

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="a forest | a camel",
                    help="use '|' as the delimiter to compose separate sentences.")
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--scale", type=float, default=10)
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

with autocast('cpu' if not has_cuda else 'cuda'):
    image = pipe(prompt, guidance_scale=scale, num_inference_steps=steps)["sample"][0]
    image.save('_'.join([x.strip() for x in prompt.split('|')]) + '.png')
