from setuptools import setup, find_packages

setup(
    name="adapted-sequential-diffusion",
    version="0.1.0",
    description=(
        "Adapted sequential diffusion for SPX implied-volatility surface "
        "simulation and option hedging"
    ),
    url="https://github.com/yinbinhan/volatility-surface-simulation",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "torchvision",
        "einops>=0.6.0",
        "ema-pytorch>=0.2.0",
        "accelerate>=0.20.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "packaging",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "Pillow",
        "tqdm",
        "tensorboard",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.10",
)
