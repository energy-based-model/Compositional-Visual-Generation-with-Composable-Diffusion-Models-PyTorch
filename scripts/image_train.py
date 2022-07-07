"""
Train a diffusion model on images.
"""
import os
import argparse
import json

from composable_diffusion import dist_util, logger
from composable_diffusion.image_datasets import load_data
from composable_diffusion.resample import create_named_schedule_sampler
from composable_diffusion.model_creation import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from composable_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()
    use_captions = args.use_captions

    dist_util.setup_dist()
    log_folder = f'./logs_{args.dataset}_{args.image_size}'
    os.makedirs(log_folder, exist_ok=True)
    logger.configure(log_folder)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    json.dump(args_to_dict(args, model_and_diffusion_defaults().keys()),
              open(os.path.join(log_folder, 'arguments.json'), "w"), sort_keys=True, indent=4)

    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        root=args.data_dir,
        split='train',
        dataset_type=args.dataset,
        batch_size=args.batch_size,
        image_size=args.image_size,
        use_captions=use_captions,
        deterministic=False,
        random_crop=False,
        random_flip=False
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        dataset=args.dataset,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset="",
        use_captions=False
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
