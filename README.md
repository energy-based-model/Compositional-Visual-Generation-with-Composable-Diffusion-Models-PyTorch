# Composable Diffusion

### [Project Page](https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/) | [Paper](https://arxiv.org/pdf/2206.01714.pdf) | [Demo](https://huggingface.co/spaces/Shuang59/Composable-Diffusion)
[![][colab]][composable-glide]



This is the official codebase for **Compositional Visual Generation with Composable Diffusion Models**.

[Compositional Visual Generation with Composable Diffusion Models](https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/)
    <br>
    [Nan Liu](https://nanliu.io) <sup>1*</sup>,
    [Shuang Li](https://people.csail.mit.edu/lishuang) <sup>2*</sup>,
    [Yilun Du](https://yilundu.github.io) <sup>2*</sup>,
    [Antonio Torralba](https://groups.csail.mit.edu/vision/torralbalab/) <sup>2</sup>,
    [Joshua B. Tenenbaum](https://mitibmwatsonailab.mit.edu/people/joshua-tenenbaum/) <sup>2</sup>
    <br>
    <sup>*</sup> Equal Contributation
    <br>
    <sup>1</sup>UIUC, <sup>2</sup>MIT CSAIL
    <br>
    [ECCV 2022](https://arxiv.org/pdf/2206.01714.pdf)



--------------------------------------------------------------------------------------------------------
* The codebase is built upon [GLIDE](https://github.com/openai/glide-text2im) and [Improved-Diffusion](https://github.com/openai/improved-diffusion).
* This codebase provides both training and inference code.

--------------------------------------------------------------------------------------------------------
[colab]: <https://colab.research.google.com/assets/colab-badge.svg>
[composable-glide]: <https://colab.research.google.com/github/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch/blob/main/notebooks/compose_glide.ipynb>


## Setup

Run following to create a conda environment, and activate it:
```
conda create -n compose_diff python=3.8
conda activate compose_diff
```
To install this package, clone this repository and then run:

```
pip install -e .
```
--------------------------------------------------------------------------------------------------------

## Training
* We follow the same manner as  [Improved-Diffusion](https://github.com/openai/improved-diffusion) for training.

To train a model on **CLEVR Objects**, we need to decide some hyperparameters as follows:
```
MODEL_FLAGS="--image_size 128 --num_channels 192 --num_res_blocks 2 --learn_sigma True --use_scale_shift_norm False --num_classes 2  --raw_unet True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule squaredcos_cap_v2 --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-5 --batch_size 16 --use_kl False --schedule_sampler loss-second-moment --microbatch -1"
```
Then, we run training script as such:
```
python scripts/image_train.py --data_dir ./dataset/ --dataset clevr_pos $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAG
```

Similarly, we use following commands to train a model on **CLEVR Relations**:
```
MODEL_FLAGS="--image_size 128 --num_channels 192 --num_res_blocks 2 --learn_sigma True --use_scale_shift_norm False --num_classes 4,3,9,3,3,7 --raw_unet True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule squaredcos_cap_v2 --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-5 --batch_size 16 --use_kl False --schedule_sampler loss-second-moment --microbatch -1"
python scripts/image_train.py --data_dir ./dataset/ --dataset clevr_rel $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```
--------------------------------------------------------------------------------------------------------

## Inference

To sample from the model, we run the inference script for **CLEVR Objects**:
```
MODEL_FLAGS="--image_size 128 --num_channels 192 --num_res_blocks 2 --learn_sigma False --use_scale_shift_norm False --num_classes 2 --dataset clevr_pos --raw_unet True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule squaredcos_cap_v2 --rescale_learned_sigmas False --rescale_timesteps False"
python scripts/image_sample_compose_clevr_pos.py $MODEL_FLAGS $DIFFUSION_FLAGS --ckpt_path $YOUR_CHECKPOINT_PATH
```

Similarly, run following command to sample from Model trained on **CLEVR Relations**:
```
MODEL_FLAGS="--image_size 128 --num_channels 192 --num_res_blocks 2 --learn_sigma True --use_scale_shift_norm False --num_classes 4,3,9,3,3,7 --raw_unet True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule squaredcos_cap_v2 --rescale_learned_sigmas False --rescale_timesteps False"
python scripts/image_sample_compose_clevr_rel.py $MODEL_FLAGS $DIFFUSION_FLAGS --ckpt_path $YOUR_CHECKPOINT_PATH
```

--------------------------------------------------------------------------------------------------------

For detailed usage examples, see the [notebooks](notebooks) directory.
 * The [composable_glide](notebooks/compose_glide.ipynb) [![][colab]][composable-glide] notebook shows how to compose GLIDE for image generation.


For python inference scripts to run on your own GPUs.
    ```
    python scripts/image_sample_compose_glide.py
    ``` 
 