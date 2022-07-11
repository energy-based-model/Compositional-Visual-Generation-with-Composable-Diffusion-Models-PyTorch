from setuptools import setup

setup(
    name="composable-diffusion",
    packages=[
        "composable_diffusion",
        "composable_diffusion.clip",
        "composable_diffusion.tokenizer",
    ],
    package_data={
        "composable_diffusion.tokenizer": [
            "bpe_simple_vocab_16e6.txt.gz",
            "encoder.json.gz",
            "vocab.bpe.gz",
        ],
        "composable_diffusion.clip": ["config.yaml"],
    },
    install_requires=[
        "Pillow",
        "attrs",
        "torch",
        "filelock",
        "requests",
        "tqdm",
        "ftfy",
        "regex",
        "numpy",
        "blobfile",
        "torchvision"
    ],
    author="nanliu",
)
