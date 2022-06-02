# Composable Diffusion

### [Project Page](https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/) | [Paper](https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/)
[![][colab]][composable-glide]

This is the official codebase for **Compositional Generation using Diffusion Models**.
The codebase is heavily built upon [GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://github.com/openai/glide-text2im).

[Compositional Visual Generation with Composable Diffusion Models](https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/)
    <br>
    [Nan Liu](https://nanliu.io) <sup>1*</sup>,
    [Shuang Li](https://people.csail.mit.edu/lishuang) <sup>2*</sup>,
    [Yilun Du](https://yilundu.github.io) <sup>2*</sup>,
    [Antonio Torralba](https://groups.csail.mit.edu/vision/torralbalab/) <sup>2</sup>,
    [Joshua B. Tenenbaum](https://mitibmwatsonailab.mit.edu/people/joshua-tenenbaum/) <sup>2</sup>,
    <br>
    <sup>*</sup> Equal Contributation
    <br>
    <sup>1</sup>UIUC, <sup>2</sup>MIT CSAIL
    <br>
    [Arxiv 2022]()

To install this package, clone this repository and then run:

```
pip install -e .
```

For detailed usage examples, see the [notebooks](notebooks) directory.

 * The [composable_glide](notebooks/compose_glide.ipynb) [![][colab]][composable-glide] notebook shows how to use GLIDE for compositional generation.

[colab]: <https://colab.research.google.com/assets/colab-badge.svg>
[composable-glide]: <https://colab.research.google.com/github/nanlliu/composable-diffusion-pytorch/blob/master/notebooks/compose_glide.ipynb>

For python inference scripts to run on your own GPUs. See the [scripts](scripts) directory.
 * The python script version of [composable_glide](notebooks/compose_glide.ipynb)
    ```
    python scripts/image_sample_compose_glide.py
    ``` 