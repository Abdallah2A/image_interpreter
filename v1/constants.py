import torch
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Constants
BASE_DIR = 'data'
WORKING_DIR = '/'
BATCH_SIZE = 1024
EMBED_DIM = 256
HIDDEN_SIZE = 512
NUM_EPOCHS = 30
LEARNING_RATE = 3e-4
BEAM_WIDTH = 3
EXPECTED_FEATURE_DIM = 2048
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
