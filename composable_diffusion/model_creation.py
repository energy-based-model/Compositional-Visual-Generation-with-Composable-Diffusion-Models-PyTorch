import argparse

from composable_diffusion import gaussian_diffusion as gd
from composable_diffusion.gaussian_diffusion import get_named_beta_schedule
from composable_diffusion.respace import SpacedDiffusion, space_timesteps
from composable_diffusion.text2im_model import (
    InpaintText2ImUNet,
    SuperResInpaintText2ImUnet,
    SuperResText2ImUNet,
    Text2ImUNet,
)

from composable_diffusion.unet import UNetModel, SuperResUNetModel
from composable_diffusion.tokenizer.bpe import get_encoder


def model_and_diffusion_defaults():
    return dict(
        image_size=64,
        num_channels=192,
        num_res_blocks=3,
        channel_mult="",
        num_heads=1,
        num_head_channels=64,
        num_heads_upsample=-1,
        attention_resolutions="32,16,8",
        dropout=0.1,
        text_ctx=128,
        xf_width=512,
        xf_layers=16,
        xf_heads=8,
        xf_final_ln=True,
        xf_padding=True,
        diffusion_steps=1000,
        noise_schedule="squaredcos_cap_v2",
        timestep_respacing="",
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=False,
        cache_text_emb=False,
        inpaint=False,
        super_res=False,
        raw_unet=False,
        learn_sigma=False,
        use_kl=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        num_classes="",
        dataset=""
    )


def model_and_diffusion_defaults_upsampler():
    result = model_and_diffusion_defaults()
    result.update(
        dict(
            image_size=256,
            num_res_blocks=2,
            noise_schedule="linear",
            super_res=True,
        )
    )
    return result


def create_model_and_diffusion(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    text_ctx,
    xf_width,
    xf_layers,
    xf_heads,
    xf_final_ln,
    xf_padding,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    learn_sigma,
    use_kl,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    cache_text_emb,
    inpaint,
    super_res,
    raw_unet,
    num_classes,
    dataset
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        text_ctx=text_ctx,
        xf_width=xf_width,
        xf_layers=xf_layers,
        xf_heads=xf_heads,
        xf_final_ln=xf_final_ln,
        xf_padding=xf_padding,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        cache_text_emb=cache_text_emb,
        inpaint=inpaint,
        super_res=super_res,
        raw_unet=raw_unet,
        num_classes=num_classes,
        dataset=dataset
    )
    diffusion = create_gaussian_diffusion(
        learn_sigma=learn_sigma,
        use_kl=use_kl,
        steps=diffusion_steps,
        noise_schedule=noise_schedule,
        timestep_respacing=timestep_respacing,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas
    )
    return model, diffusion


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult,
    attention_resolutions,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    text_ctx,
    xf_width,
    xf_layers,
    xf_heads,
    xf_final_ln,
    xf_padding,
    resblock_updown,
    use_fp16,
    cache_text_emb,
    inpaint,
    super_res,
    raw_unet,
    num_classes,
    dataset
):
    if channel_mult == "":
        if image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))
        assert 2 ** (len(channel_mult) + 2) == image_size

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    if raw_unet:
        if super_res:
            model = SuperResUNetModel
        else:
            model = UNetModel
        return model(
            in_channels=3,
            model_channels=num_channels,
            out_channels=6,
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=num_classes,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            encoder_channels=None,
            dataset=dataset
        )
    else:
        if inpaint and super_res:
            model_cls = SuperResInpaintText2ImUnet
        elif inpaint:
            model_cls = InpaintText2ImUNet
        elif super_res:
            model_cls = SuperResText2ImUNet
        else:
            model_cls = Text2ImUNet
        return model_cls(
            text_ctx=text_ctx,
            xf_width=xf_width,
            xf_layers=xf_layers,
            xf_heads=xf_heads,
            xf_final_ln=xf_final_ln,
            tokenizer=get_encoder(),
            xf_padding=xf_padding,
            in_channels=3,
            model_channels=num_channels,
            out_channels=6,
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            cache_text_emb=cache_text_emb,
        )


def create_gaussian_diffusion(
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="squaredcos_cap_v2",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
