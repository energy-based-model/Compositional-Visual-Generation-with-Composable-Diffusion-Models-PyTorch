# Composable Diffusion

### [Project Page](https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/) | [Paper](https://arxiv.org/pdf/2206.01714.pdf)
[![][colab]][composable-glide]

Try Replicate web demo here [![Replicate](https://replicate.com/energy-based-model/compositional-vsual-generation-with-composable-diffusion-models-pytorch/badge)](https://replicate.com/energy-based-model/compositional-vsual-generation-with-composable-diffusion-models-pytorch)




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
    [Arxiv 2022](https://arxiv.org/pdf/2206.01714.pdf)



--------------------------------------------------------------------------------------------------------
* This code is the basic version of our paper. We will release the final version soon.
* The codebase is built upon [GLIDE](https://github.com/openai/glide-text2im).




--------------------------------------------------------------------------------------------------------
To install this package, clone this repository and then run:

```
pip install -e .
```


[colab]: <https://colab.research.google.com/assets/colab-badge.svg>
[composable-glide]: <https://colab.research.google.com/github/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch/blob/main/notebooks/compose_glide.ipynb>



For detailed usage examples, see the [notebooks](notebooks) directory.
 * The [composable_glide](notebooks/compose_glide.ipynb) [![][colab]][composable-glide] notebook shows how to compose GLIDE for image generation.


For python inference scripts to run on your own GPUs.
    ```
    python scripts/image_sample_compose_glide.py
    ``` 
 
