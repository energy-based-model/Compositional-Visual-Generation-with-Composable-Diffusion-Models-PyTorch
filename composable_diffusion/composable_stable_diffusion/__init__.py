"""
A codebase for performing model inference with a text-conditional diffusion model.
"""

from dataclasses import dataclass
from typing import List, Union

import numpy as np

import PIL

from diffusers.utils import BaseOutput


@dataclass
class StableDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.
    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
        nsfw_content_detected (`List[bool]`)
            List of flags denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: List[bool]