import torch
import shutil
import os

# -------------------------------
# Experiment Tracking & Reproducibility Config
# -------------------------------

# Experiment version or ID (update for each major run)
EXPERIMENT_ID = "exp1"

# Project root (absolute path)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

ALPHA = 0.95  # Weight for positive class in Focal Loss (0 < alpha < 1)
# Data and paths (absolute)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LOG_DIR = os.path.join(PROJECT_ROOT, "experiments", "runs", EXPERIMENT_ID)
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "experiments", f"{EXPERIMENT_ID}_best_model.pt")
SPLIT_DIR = os.path.join(PROJECT_ROOT, "experiments", "splits")
PREDICTIONS_PATH = os.path.join(PROJECT_ROOT, "experiments", f"{EXPERIMENT_ID}_test_predictions.csv")
REGISTRY_PATH = os.path.join(PROJECT_ROOT, "experiments", "experiment_registry.csv")

# Model hyperparameters
HIDDEN_DIM = 128
NUM_LAYERS = 4
DROPOUT = 0.3
GNN_TYPE = "gine"  # 'gine' or 'nnconv'
NORM_TYPE = "batch"  # 'layer' or 'batch'
MLP_HIDDEN_DIM = 128

# Training hyperparameters
BATCH_SIZE = 64
NUM_EPOCHS = 100
PATIENCE = 10  # Early stopping patience
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GAMMA = 3.0  # Focal loss gamma
SEED = 42

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cross-validation
N_SPLITS = 5  # Number of folds for KFold cross-validation
SHUFFLE = True
RANDOM_STATE = SEED

# -------------------------------
# Utility: Save a snapshot of this config for each experiment run
# -------------------------------
def save_config_snapshot(destination_dir=None):
    """
    Save a copy of this config.py to the experiment directory for reproducibility.
    """
    src_path = os.path.abspath(__file__)
    if destination_dir is None:
        destination_dir = LOG_DIR
    os.makedirs(destination_dir, exist_ok=True)
    dst_path = os.path.join(destination_dir, "config_snapshot.py")
    shutil.copy2(src_path, dst_path)
    print(f"Config snapshot saved to {dst_path}")
