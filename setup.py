"""Setup script for the package"""
from setuptools import setup, find_packages

setup(
    name="small-language-model",
    version="0.1.0",
    description="A small transformer-based language model built from scratch",
    author="Sudeera Dilshan",
    author_email="dilshanrgs31@gmail.com",
    url="git@github.com:SudeeraDilshan/A-small-transformer-based-language-model-built-from-scratch.git",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "datasets>=2.10.0",
        "tokenizers>=0.13.0",
        "transformers>=4.30.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
        ],
        "wandb": [
            "wandb>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "slm=scripts.main:main",
        ],
    },
)
