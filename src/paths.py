import os
from src import config

# Data directories
RAW_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
PROCESSED_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))

# Experiments and logs
EXPERIMENTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments"))
SPLIT_DIR = os.path.abspath(os.path.join(EXPERIMENTS_DIR, "splits")) # Keep splits separate for now
REGISTRY_PATH = os.path.abspath(os.path.join(EXPERIMENTS_DIR, "experiment_registry.csv")) # Keep registry at top level
# LOG_DIR, CHECKPOINT_PATH, PREDICTIONS_PATH removed - paths generated dynamically per experiment

def experiment_dir(experiment_id):
    """Return the path to the directory for a specific experiment run."""
    return os.path.join(EXPERIMENTS_DIR, str(experiment_id))

def raw_scenario_path(scenario_id):
    """Return the path to a raw HDF5 scenario file."""
    return os.path.join(RAW_DATA_DIR, f"scenario_{scenario_id}.h5")

def processed_scenario_path(scenario_id):
    """Return the path to a processed .pt scenario file."""
    return os.path.join(PROCESSED_DATA_DIR, f"scenario_{scenario_id}.pt")

def split_index_path(split_type, fold=None):
    """
    Return the path to a split index file.
    split_type: 'train', 'val', or 'test'
    fold: integer fold number (optional)
    """
    if fold is not None:
        return os.path.join(SPLIT_DIR, f"{split_type}_idx_fold{fold}.npy")
    else:
        return os.path.join(SPLIT_DIR, f"{split_type}_idx.npy")

def checkpoint_path(experiment_id=None, fold=None):
    """
    Return the path to a model checkpoint file.
    Uses the provided experiment_id if given, otherwise defaults to config.EXPERIMENT_ID.
    """
    current_experiment_id = experiment_id if experiment_id is not None else config.EXPERIMENT_ID
    filename_base = f"{current_experiment_id}_best_model"
    if fold is not None:
        filename = f"{filename_base}_fold{fold}.pt"
    else:
        filename = f"{filename_base}.pt"
    exp_dir = experiment_dir(current_experiment_id)
    # Directory creation moved to train.py
    return os.path.join(exp_dir, filename)

def predictions_path(experiment_id=None, fold=None):
    """
    Return the path to a predictions CSV file.
    Uses the provided experiment_id if given, otherwise defaults to config.EXPERIMENT_ID.
    """
    current_experiment_id = experiment_id if experiment_id is not None else config.EXPERIMENT_ID
    filename_base = f"{current_experiment_id}_test_predictions"
    if fold is not None:
        filename = f"{filename_base}_fold{fold}.csv"
    else:
        filename = f"{filename_base}.csv"
    exp_dir = experiment_dir(current_experiment_id)
    # Directory creation moved to train.py
    return os.path.join(exp_dir, filename)


def optimal_threshold_path(experiment_id):
    """
    Return the path to the optimal threshold file.
    """
    filename = "optimal_threshold.txt"
    exp_dir = experiment_dir(experiment_id)
    # Directory creation moved to train.py
    return os.path.join(exp_dir, filename)

def ensure_dirs():
    """Ensure base directories exist."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    os.makedirs(SPLIT_DIR, exist_ok=True) # Ensure the global split directory exists