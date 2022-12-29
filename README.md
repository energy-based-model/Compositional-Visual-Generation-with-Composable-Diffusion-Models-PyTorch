# Composable Diffusion
## We propose to use conjunction and negation (negative prompts) operators for compositional generation with conditional diffusion models (i.e., Stable Diffusion, Point-E, etc).

### [Project Page](https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/) | [Paper](https://arxiv.org/pdf/2206.01714.pdf) | [Google Colab][composable-demo] | [Huggingface][huggingface-demo]
[![][colab]][composable-demo] [![][huggingface]][huggingface-demo]

<hr>

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
    [ECCV 2022](https://arxiv.org/pdf/2206.01714.pdf) / [MIT News](https://news.mit.edu/2022/ai-system-makes-models-like-dall-e-2-more-creative-0908) / [MIT CSAIL News](https://www.csail.mit.edu/news/ai-system-makes-models-dall-e-2-more-creative)

[colab]: <https://colab.research.google.com/assets/colab-badge.svg>
[huggingface]: <https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue>
[composable-demo]: <https://colab.research.google.com/github/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch/blob/main/notebooks/demo.ipynb>
[huggingface-demo]: <https://huggingface.co/spaces/Shuang59/Composable-Diffusion>
<hr>

## Composed 2D Image Results using **[Stable-Diffusion](https://github.com/CompVis/stable-diffusion)**.
![](samples/example1_A.gif)  |  ![](samples/example1_N.gif)
:-------------------------:|:-------------------------:

| Image | Positive Prompts (AND Operator) | Negative Prompts (NOT Operator) |
| --------------- | --------------- | --------------- |
| ```Left``` | ```["A stone castle surrounded by lakes and trees, fantasy, wallpaper, concept art, extremely detailed", "Black and white"]``` | ```None``` |
| ```Right``` | ```["A stone castle surrounded by lakes and trees, fantasy, wallpaper, concept art, extremely detailed"]``` | ```["Black and white"]``` |


![](samples/example2_A.gif)  |  ![](samples/example2_N.gif)
:-------------------------:|:-------------------------:

| Image | Positive Prompts (AND Operator) | Negative Prompts (NOT Operator) |
| --------------- | --------------- | --------------- |
| ```Left``` | ```["mystical trees", "A magical pond", "Dark"]``` | ```None``` |
| ```Right``` | ```["mystical trees", "A magical pond"]``` | ```["Dark"]``` |


1. Samples generated by [Stable-Diffusion](https://github.com/CompVis/stable-diffusion) using our compositional generation operator.
2. More discussions and results about our proposed methods can be found in [Reddit Post 1](https://www.reddit.com/r/StableDiffusion/comments/xwplfv/and_prompt_combinations_just_landed_in/), [Reddit Post 2](https://www.reddit.com/r/StableDiffusion/comments/xf5jow/compositional_diffusion/) and [Reddit Post 3](https://www.reddit.com/r/StableDiffusion/comments/xoq7ik/composable_diffusion_a_new_development_to_greatly/)!
3. Some prompts are borrowed from [Lexica](https://lexica.art/)!

<hr>

## Composed 3D Mesh Results using **[Point-E](https://github.com/openai/point-e)**.

![](samples/a%20green%20avocado_a%20chair.gif)| ![](samples/a%20chair_not%20chair%20legs.gif)  | ![](samples/a%20toilet_a%20chair.gif) 
:-------------------------:|:----------------------------------------------:|:-------------------------:
```A green avocado AND A chair```|        ```A chair AND NOT Chair legs```        | ```A toilet AND A chair```
![](samples/a%20couch_a%20boat.gif)| ![](samples/a%20monitor_a%20brown%20couch.gif) | ![](samples/a%20chair_a%20cake.gif)
```A couch AND A boat``` |       ```A monitor AND A brown couch```        | ```A chair AND A cake```

## **News**

--------------------------------------------------------------------------------------------------------
* <b>12/22/22</b>: Now you can use our code to apply compositional operator (AND) to **[Point-E](https://github.com/openai/point-e)**!
* <b>12/13/22</b>: ```stabilityai/stable-diffusion-2-1-base``` and other updated versions can now be used for compositional generation. (see [here](https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch/blob/main/scripts/image_sample_compose_stable_diffusion.py)!)
* <b>10/10/22</b>: Our proposed operators have been added into [stable-diffusion-webui-conjunction](https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/c26732fbee2a57e621ac22bf70decf7496daa4cd)!
* <b>09/08/22</b>: Our paper is on [MIT News](https://news.mit.edu/2022/ai-system-makes-models-like-dall-e-2-more-creative-0908) and [MIT CSAIL News](https://www.csail.mit.edu/news/ai-system-makes-models-dall-e-2-more-creative)!
* Now you can try to use compose **[Stable-Diffusion](https://github.com/CompVis/stable-diffusion)** Model using our [![][huggingface]][huggingface-demo] or [![][colab]][composable-demo] to sample 512x512 images.
--------------------------------------------------------------------------------------------------------
* The codebase is built upon [GLIDE](https://github.com/openai/glide-text2im) and [Improved-Diffusion](https://github.com/openai/improved-diffusion).
* This codebase provides both training and inference code.
* **The codebase can be used to train text-conditioned diffusion model in a similar manner as [GLIDE](https://github.com/openai/glide-text2im).**

--------------------------------------------------------------------------------------------------------

## Setup

Run following to create a conda environment, and activate it:
```
conda create -n compose_diff python=3.8
conda activate compose_diff
```
To install this package, clone this repository and then run:

```
pip install -e .
pip install diffusers==0.10.2
pip install open3d==0.16.0
```
--------------------------------------------------------------------------------------------------------
## Inference

### Google Colab 
The [demo](notebooks/demo.ipynb) [![][colab]][composable-demo] notebook shows how to compose natural language descriptions, and CLEVR objects for image generation.

### Python
Compose natural language descriptions to generate 3D mesh using [Point-E](https://github.com/openai/point-e):
```
python scripts/txt2pointclouds_compose_pointe.py --prompts "a cake" "a house" --weights 3 3
```

Compose natural language descriptions using [Stable-Diffusion](https://github.com/CompVis/stable-diffusion):
```
# Conjunction (AND) by specifying positive weights
# weights can be adjusted, otherwise will be the same as scale
python scripts/image_sample_compose_stable_diffusion.py --prompts "mystical trees" "A magical pond" "dark" --weights 7.5 7.5 7.5 --scale 7.5 --steps 50 --seed 2
```
```
# NEGATION (NOT) by specifying negative weights
python scripts/image_sample_compose_stable_diffusion.py --prompts "mystical trees" "A magical pond" "dark" --weights 7.5 7.5 -7.5 --scale 7.5 --steps 50 --seed 2
```

Compose natural language descriptions using pretrained [GLIDE](https://github.com/openai/glide-text2im):
```
# Conjunction (AND) 
python scripts/image_sample_compose_glide.py --prompts "a camel" "a forest" --weights 7.5 7.5 --steps 100
```

Compose objects:
```
# Conjunction (AND) 
MODEL_FLAGS="--image_size 128 --num_channels 192 --num_res_blocks 2 --learn_sigma False --use_scale_shift_norm False --num_classes 2 --dataset clevr_pos --raw_unet True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule squaredcos_cap_v2 --rescale_learned_sigmas False --rescale_timesteps False"
python scripts/image_sample_compose_clevr_pos.py $MODEL_FLAGS $DIFFUSION_FLAGS --ckpt_path $YOUR_CHECKPOINT_PATH
```

Compose objects relational descriptions:
```
# Conjunction (AND) 
MODEL_FLAGS="--image_size 128 --num_channels 192 --num_res_blocks 2 --learn_sigma True --use_scale_shift_norm False --num_classes 4,3,9,3,3,7 --raw_unet True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule squaredcos_cap_v2 --rescale_learned_sigmas False --rescale_timesteps False"
python scripts/image_sample_compose_clevr_rel.py $MODEL_FLAGS $DIFFUSION_FLAGS --ckpt_path $YOUR_CHECKPOINT_PATH
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

To train a text-conditioned GLIDE model, we also provide code for training on **MS-COCO** dataset. \
Firstly, specify the image root directory path and corresponding json file for captions
in [image_dataset](https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch/blob/main/composable_diffusion/image_datasets.py) file.\
Then, we can use following command example to train a model on MS-COCO captions:
```
MODEL_FLAGS="--image_size 128 --num_channels 192 --num_res_blocks 2 --learn_sigma True --use_scale_shift_norm False"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule squaredcos_cap_v2 --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-5 --batch_size 16 --use_kl False --schedule_sampler loss-second-moment --microbatch -1"
python scripts/image_train.py --data_dir ./dataset/ --dataset coco $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

--------------------------------------------------------------------------------------------------------

## Dataset
Training datasets for both **CLEVR Objects** and **CLEVR Relations** will be downloaded automatically
when running the script above.

If you need to manually download, the datasets used for training our models can be found at:

| Dataset | Link | 
| :---: | :---: | 
| CLEVR Objects | https://www.dropbox.com/s/5zj9ci24ofo949l/clevr_pos_data_128_30000.npz?dl=0
| CLEVR Relations | https://www.dropbox.com/s/urd3zgimz72aofo/clevr_training_data_128.npz?dl=0
--------------------------------------------------------------------------------------------------------

## Citing our Paper

If you find our code useful for your research, please consider citing 

``` 
@article{liu2022compositional,
  title={Compositional Visual Generation with Composable Diffusion Models},
  author={Liu, Nan and Li, Shuang and Du, Yilun and Torralba, Antonio and Tenenbaum, Joshua B},
  journal={arXiv preprint arXiv:2206.01714},
  year={2022}
}
```
