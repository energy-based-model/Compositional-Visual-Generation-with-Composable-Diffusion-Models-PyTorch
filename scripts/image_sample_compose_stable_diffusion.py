import torch as th
import numpy as np
import torchvision.utils as tvu

from torch import autocast
from PIL import Image
from diffusers import StableDiffusionPipeline

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default="a forest | a camel",
                    help="use '|' as the delimiter to compose separate sentences.")
parser.add_argument("--steps", default=50)
parser.add_argument("--scale", default=10)
args = parser.parse_args()


has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')

prompt = args.prompt
scale = args.scale
steps = args.steps

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token='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'
).to(device)

with autocast('cpu' if not has_cuda else 'cuda'):
    image = pipe(prompt, guidance_scale=scale, num_inference_steps=steps)["sample"][0]
    image.save('_'.join([x.strip() for x in prompt.split('|')]) + '.png')
