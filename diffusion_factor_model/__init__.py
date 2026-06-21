"""
Diffusion Factor Model: A diffusion-based framework for financial factor modeling
"""

from diffusion_factor_model.diffusion_factor_model import (
    Unet,
    GaussianDiffusion,
    SequentialGaussianDiffusion,
    ConditionalTransformer,
    Trainer,
    GaussianLatentSampler2D_Finance
)

__version__ = "0.1.0"


from diffusion_factor_model.fine_tuning import (
    LoRALinear,
    OnlineDDPMLoRAFineTuner,
    FineTuneStats,
    ArbitrageValidator,
    make_arbitrage_reward_fn,
)
