"""
Configuration for Diffusion Factor Model
"""

import os
import torch
import numpy as np

# Project directory paths - Using relative paths from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "model_results")
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Experiment naming
EXP_PREFIX = "dfm"  # Prefix for experiment IDs

# Core settings
SEED = 3407
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model parameters (legacy UNet support)
MODEL_DIM = 256            # Base dimension for U-Net
MODEL_CHANNELS = 1        # Number of channels in input data
MODEL_FILTER_SIZE = 7     # Filter size for convolutions

# Transformer parameters
TRANSFORMER_DIM = 256
TRANSFORMER_LAYERS = 8
TRANSFORMER_HEADS = 4
TRANSFORMER_FF_MULT = 2
TRANSFORMER_DROPOUT = 0.1
USE_BOS_TOKEN = False
USE_ALIBI = False
ALIBI_SLOPE = 1.0
FIRST_TOKEN_BIAS = 0.0

# Dimension multipliers for different input sizes
DIM_MULTS_LARGE = (1, 2, 4, 16)     # For inputs where min_dim >= 32
DIM_MULTS_MEDIUM = (1, 2, 4, 8)       # For inputs where min_dim >= 16
DIM_MULTS_SMALL = (1, 2, 4)           # For inputs where min_dim >= 8
DIM_MULTS_TINY = (1, 2)            # For inputs where min_dim >= 4
DIM_MULTS_MINIMAL = (1,)           # For very small inputs

# Diffusion parameters
TIMESTEPS = 200
SAMPLING_TIMESTEPS = 20   # Number of steps used for DDIM sampling (set >= TIMESTEPS to disable DDIM)
DDIM_ETA = 0.0            # Noise weight for DDIM (0.0 makes sampling deterministic)
OBJECTIVE = 'pred_x0'
BETA_SCHEDULE = 'cosine'
AUTO_NORMALIZE = False

# Training parameters
BATCH_SIZE = 256
LEARNING_RATE = 7e-5
EPOCHS = 3000
WEIGHT_DECAY = 0.01
USE_COSINE_SCHEDULER = True
USE_WARM_UP = True
WARMUP_STEPS = 1600
COSINE_CYCLE_LENGTH = 1400  # T_0 (initial cycle length)
T_MULT = 1                 # T_mult for scheduler
COSINE_STEPS = 1400         # Cosine annealing steps (same as T_0 by default)
COSINE_LR_MIN = 1e-06      # ETA_MIN
GRADIENT_ACCUMULATION = 1
EMA_DECAY = 0.995
SPLIT_BATCHES = False
SAVE_INTERVAL = 100       # Save checkpoint every N epochs

# Sampling parameters
SAMPLE_BATCHES = 8      # Number of batches to sample
SAMPLES_PER_BATCH = 128   # Number of samples per batch
SAVE_TIMESTEPS = None     # For sequential sampling, defaults to final output only
SAMPLE_WINDOW_START = 0   # Default start index for sequential sampling windows
SAMPLE_WINDOW_LENGTH = None # Number of sequential indices to generate by default (capped by sequence length)

# Mixed precision settings
USE_AMP = True            # Mixed precision training

# Data parameters
TRAIN_SAMPLES = 2**11     # Number of samples to use for training

# File naming and paths
def get_experiment_id(seed=None, num_samples=None):
    """Generate a unique experiment identifier"""
    if seed is None:
        seed = SEED
    
    if num_samples is None:
        num_samples = TRAIN_SAMPLES
    
    return (f"finance_2D_dim{MODEL_DIM}_"
           f"latent{MODEL_DIM}_"
           f"Tmax{COSINE_CYCLE_LENGTH}_"
           f"etamin{COSINE_LR_MIN}_"
           f"batchsize{BATCH_SIZE}_"
           f"samples{int(np.log2(num_samples))}_"
           f"seed{seed}")

def set_seed(seed=None):
    """Set random seed for reproducibility"""
    if seed is None:
        seed = SEED
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    return seed

def get_model_path(exp_id=None):
    """Get the full path to the model directory for an experiment"""
    if exp_id is None:
        exp_id = get_experiment_id()
    return os.path.join(MODELS_DIR, exp_id)

def get_samples_path(exp_id=None):
    """Get the full path to the samples directory for an experiment"""
    if exp_id is None:
        exp_id = get_experiment_id()
    return os.path.join(SAMPLES_DIR, exp_id) 
