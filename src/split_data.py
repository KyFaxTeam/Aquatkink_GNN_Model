import os
import numpy as np
from sklearn.model_selection import train_test_split
from src import paths

def split_train_val_test_and_save(
    dataset,
    test_size=0.2,
    val_fraction=0.15, # Fraction of the *initial* train set
    split_dir=None,
    random_state=42,
    stratify=None # Note: Stratification might be tricky across two splits
):
    """
    Splits the dataset into train, validation, and test indices and saves them.

    Performs a two-stage split:
    1. Splits the full dataset into initial train and test sets.
    2. Splits the initial train set into final train and validation sets.

    Args:
        dataset: Dataset object (must support len()).
        test_size: Fraction or int for the test set size (from the full dataset).
        val_fraction: Fraction of the *initial* train set to use for validation.
        split_dir: Directory to save split indices. Defaults to paths.SPLIT_DIR.
        random_state: Random seed for reproducibility.
        stratify: Optional array for stratified splitting (applied to the first split).
                  If stratification is needed for the second split, it requires
                  mapping the stratify array to the initial train indices.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: train_idx, val_idx, test_idx
    """
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    if split_dir is None:
        split_dir = paths.SPLIT_DIR

    # --- First Split: Train / Test ---
    # Stratification is applied here if provided
    initial_train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify # Stratify based on the full dataset labels if provided
    )
    print(f"Initial split: {len(initial_train_idx)} train, {len(test_idx)} test samples")

    # --- Second Split: Train / Validation (from initial_train_idx) ---
    # Handle stratification for the second split if needed (more complex)
    # For now, we assume stratification is either not needed here or handled by stratifying the first split
    # If stratify was provided, we might need to map it to the initial_train_idx
    stratify_val = None
    if stratify is not None:
        # Simple approach: use the subset of stratify corresponding to initial_train_idx
        # This assumes the stratify array aligns with the original indices
        try:
            stratify_val = stratify[initial_train_idx]
        except IndexError:
             print("Warning: Could not map stratify array to initial train indices for validation split. Proceeding without stratification for val split.")
             stratify_val = None # Fallback if mapping fails

    if len(initial_train_idx) == 0:
         raise ValueError("Initial training set is empty after train/test split.")

    # Adjust val_fraction if initial_train_idx is too small
    if int(val_fraction * len(initial_train_idx)) < 1 and len(initial_train_idx) > 0:
        print(f"Warning: val_fraction {val_fraction} results in < 1 sample for validation from initial train size {len(initial_train_idx)}. Using 1 validation sample.")
        val_samples = 1
    elif len(initial_train_idx) == 0:
         val_samples = 0 # Should be caught by the ValueError above, but defensive check
    else:
        val_samples = val_fraction # Use fraction directly for train_test_split

    if len(initial_train_idx) <= 1 and val_samples > 0:
         print("Warning: Only one sample in initial_train_idx, cannot create validation split.")
         final_train_idx = initial_train_idx
         val_idx = np.array([], dtype=initial_train_idx.dtype)
    elif val_samples == 0:
         final_train_idx = initial_train_idx
         val_idx = np.array([], dtype=initial_train_idx.dtype)
    else:
        final_train_idx, val_idx = train_test_split(
            initial_train_idx,
            test_size=val_samples, # val_fraction is relative to initial_train_idx size
            random_state=random_state, # Use the same random state for consistency
            shuffle=True,
            stratify=stratify_val # Apply mapped stratification if available
        )

    print(f"Second split: {len(final_train_idx)} final train, {len(val_idx)} validation samples")


    # --- Save Indices ---
    os.makedirs(split_dir, exist_ok=True)
    np.save(paths.split_index_path("train"), final_train_idx)
    np.save(paths.split_index_path("val"), val_idx)
    np.save(paths.split_index_path("test"), test_idx)

    print("-" * 20)
    print(f"Saved final train indices: {len(final_train_idx)} samples to {paths.split_index_path('train')}")
    print(f"Saved validation indices: {len(val_idx)} samples to {paths.split_index_path('val')}")
    print(f"Saved test indices: {len(test_idx)} samples to {paths.split_index_path('test')}")
    print("-" * 20)

    return final_train_idx, val_idx, test_idx


# --- Deprecated Functions (Commented out) ---

# def split_and_save_indices(
#     dataset,
#     test_size=0.2,
#     split_dir=None,
#     random_state=42,
#     stratify=None
# ):
#     """
#     DEPRECATED: Use split_train_val_test_and_save instead.
#     Splits the dataset into train and test indices and saves them as .npy files.
#
#     Args:
#         dataset: Dataset object (must support len()).
#         test_size: Fraction or int for test set size.
#         split_dir: Directory to save split indices.
#         random_state: Random seed for reproducibility.
#         stratify: Optional array for stratified splitting.
#     """
#     num_samples = len(dataset)
#     indices = np.arange(num_samples)
#     if split_dir is None:
#         split_dir = paths.SPLIT_DIR
#
#     train_idx, test_idx = train_test_split(
#         indices,
#         test_size=test_size,
#         random_state=random_state,
#         shuffle=True,
#         stratify=stratify
#     )
#
#     os.makedirs(split_dir, exist_ok=True)
#     np.save(paths.split_index_path("train"), train_idx)
#     np.save(paths.split_index_path("test"), test_idx)
#     print(f"Saved train indices: {len(train_idx)} samples")
#     print(f"Saved test indices: {len(test_idx)} samples")
#
#     return train_idx, test_idx

# def split_train_val_indices(
#     train_idx,
#     val_fraction=0.15,
#     split_dir=None,
#     random_state=42,
#     stratify=None
# ):
#     """
#     DEPRECATED: Use split_train_val_test_and_save instead.
#     Splits the train indices into train and validation indices and saves them as .npy files.
#
#     Args:
#         train_idx: Array of training indices.
#         val_fraction: Fraction of training set to use for validation.
#         split_dir: Directory to save split indices.
#         random_state: Random seed for reproducibility.
#         stratify: Optional array for stratified splitting.
#     """
#     if split_dir is None:
#         split_dir = paths.SPLIT_DIR
#
#     train_idx, val_idx = train_test_split(
#         train_idx,
#         test_size=val_fraction,
#         random_state=random_state,
#         shuffle=True,
#         stratify=stratify
#     )
#     os.makedirs(split_dir, exist_ok=True)
#     np.save(paths.split_index_path("train"), train_idx) # Overwrites original train!
#     np.save(paths.split_index_path("val"), val_idx)
#     print(f"Saved train indices: {len(train_idx)} samples")
#     print(f"Saved val indices: {len(val_idx)} samples")
#     return train_idx, val_idx