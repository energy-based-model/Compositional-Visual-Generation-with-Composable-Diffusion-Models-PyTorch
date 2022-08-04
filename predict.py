import tempfile
from PIL import Image
import torch as th

from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)

from cog import BasePredictor, Input, Path

from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        has_cuda = th.cuda.is_available()
        self.device = th.device("cpu" if not has_cuda else "cuda")

        timestep_respacing = 100
        self.options = model_and_diffusion_defaults()
        self.options["use_fp16"] = has_cuda
        self.options["timestep_respacing"] = str(
            timestep_respacing
        )  # use 100 diffusion steps for fast sampling
        self.model, self.diffusion = create_model_and_diffusion(**self.options)
        self.model.eval()
        if has_cuda:
            self.model.convert_to_fp16()
        self.model.to(self.device)
        self.model.load_state_dict(th.load("checkpoints/base.pt", self.device))

        print("total base parameters", sum(x.numel() for x in self.model.parameters()))

        # Create upsampler model.
        self.options_up = model_and_diffusion_defaults_upsampler()
        self.options_up["use_fp16"] = has_cuda
        self.options_up[
            "timestep_respacing"
        ] = "fast27"  # use 27 diffusion steps for very fast sampling
        self.model_up, self.diffusion_up = create_model_and_diffusion(**self.options_up)
        self.model_up.eval()
        if has_cuda:
            self.model_up.convert_to_fp16()
        self.model_up.to(self.device)
        # self.model_up.load_state_dict(load_checkpoint('upsample', self.device))
        self.model_up.load_state_dict(th.load("checkpoints/upsample.pt", self.device))
        print(
            "total upsampler parameters",
            sum(x.numel() for x in self.model_up.parameters()),
        )

    def predict(
        self,
        prompt: str = Input(
            default="a camel | a forest",
            description="Prompt for text generation. When composing  multiple sentences, using `|` as the delimiter.",
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        device = self.device
        model = self.model
        options = self.options
        diffusion = self.diffusion
        model_up = self.model_up
        options_up = self.options_up
        diffusion_up = self.diffusion_up

        batch_size = 1
        guidance_scale = 10
        # Tune this parameter to control the sharpness of 256x256 images.
        # A value of 1.0 is sharper, but sometimes results in grainy artifacts.
        upsample_temp = 0.980

        prompts = [x.strip() for x in prompt.split("|")]
        masks = [True] * len(prompts) + [False]
        # coefficients = th.tensor([0.5, 0.5], device=device).reshape(-1, 1, 1, 1)
        masks = th.tensor(masks, dtype=th.bool, device=self.device)
        # sampling function
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

        def sample_64(prompts):
            tokens_list = [model.tokenizer.encode(prompt) for prompt in prompts]
            outputs = [
                model.tokenizer.padded_tokens_and_mask(tokens, options["text_ctx"])
                for tokens in tokens_list
            ]

            cond_tokens, cond_masks = zip(*outputs)
            cond_tokens, cond_masks = list(cond_tokens), list(cond_masks)

            full_batch_size = batch_size * (len(prompts) + 1)
            uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
                [], options["text_ctx"]
            )

            # Pack the tokens together into model kwargs.
            model_kwargs = dict(
                tokens=th.tensor(cond_tokens + [uncond_tokens], device=device),
                mask=th.tensor(
                    cond_masks + [uncond_mask],
                    dtype=th.bool,
                    device=device,
                ),
            )

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
            return samples

        def upsampling_256(prompts, samples):
            tokens = model_up.tokenizer.encode("".join(prompts))
            tokens, mask = model_up.tokenizer.padded_tokens_and_mask(
                tokens, options_up["text_ctx"]
            )

            # Create the model conditioning dict.
            model_kwargs = dict(
                # Low-res image to upsample.
                low_res=((samples + 1) * 127.5).round() / 127.5 - 1,
                # Text tokens
                tokens=th.tensor([tokens] * batch_size, device=device),
                mask=th.tensor(
                    [mask] * batch_size,
                    dtype=th.bool,
                    device=device,
                ),
            )

            # Sample from the base model.
            model_up.del_cache()
            up_shape = (
                batch_size,
                3,
                options_up["image_size"],
                options_up["image_size"],
            )
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
            return up_samples

        samples = sample_64(prompts)
        upsamples = upsampling_256(prompts, samples)
        scaled = ((upsamples + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu()
        reshaped = scaled.permute(2, 0, 3, 1).reshape([upsamples.shape[2], -1, 3])

        output_path = Path(tempfile.mkdtemp()) / "output.png"
        Image.fromarray(reshaped.numpy()).save(str(output_path))
        return output_path
