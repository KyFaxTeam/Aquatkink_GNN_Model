
import numpy as np

import torch
import torch.optim as optim
import os
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score
from torch.utils.tensorboard import SummaryWriter
from src.experiment_utils import get_last_experiment_id, increment_experiment_id

from src.config import (
    DATA_DIR, HIDDEN_DIM, NUM_LAYERS, DROPOUT, GNN_TYPE, NORM_TYPE,
    MLP_HIDDEN_DIM, BATCH_SIZE, NUM_EPOCHS, PATIENCE, LEARNING_RATE, WEIGHT_DECAY, GAMMA, ALPHA, SEED,
    DEVICE
)
from src import paths
from src.datasets import WDNLeakDataset
from src.models import WDNLeakGNN
from src.utils import set_seed, compute_auc_pr, compute_mrr, compute_hits_at_k # Removed FocalLoss import
from torchvision.ops import sigmoid_focal_loss # Import torchvision focal loss

# Removed criterion parameter
def train(model, train_loader, val_loader, optimizer, scheduler, writer, experiment_id): # Added experiment_id
    best_val_auc = 0.0
    patience_counter = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            # Debug: Print label stats
            #print(f"[DEBUG] batch.y shape: {batch.y.shape}, sum: {batch.y.sum().item()}, unique: {batch.y.unique(return_counts=True)}")
            batch = batch.to(DEVICE)
            # Skip batches with no targets or no edges after processing
            if batch.y.numel() == 0 or batch.edge_index.numel() == 0:
                print(f"  [DEBUG Train Loop] Skipping batch due to empty targets or edges: {batch}")
                continue

            optimizer.zero_grad()
            # --- Debug: Check for NaNs in input ---
            if torch.isnan(batch.x).any() or torch.isinf(batch.x).any():
                print(f"[DEBUG Train Loop] NaN/Inf detected in batch.x!")
            if torch.isnan(batch.edge_attr).any() or torch.isinf(batch.edge_attr).any():
                print(f"[DEBUG Train Loop] NaN/Inf detected in batch.edge_attr!")
            # --- End Debug ---

            out = model(batch.x, batch.edge_index, batch.edge_attr)
            # Use torchvision's sigmoid_focal_loss directly
            # Use torchvision's sigmoid_focal_loss directly
            # Note: Ensure GAMMA is defined (it is in config)
            loss = sigmoid_focal_loss(out, batch.y.float(), alpha=ALPHA, gamma=GAMMA, reduction='mean')

            # --- Debug: Check for NaNs in output/loss ---
            if torch.isnan(out).any() or torch.isinf(out).any():
                print(f"[DEBUG Train Loop] NaN/Inf detected in model output (logits)!")
            # Check for NaN/Inf loss *before* backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[DEBUG Train Loop] NaN/Inf detected in calculated loss! Skipping backward pass for batch.")
                # Optionally add more debugging here about batch content if needed
                continue # Skip backward pass and optimizer step
            # --- End Debug ---

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0) # Clip gradients
            optimizer.step()
            train_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses)

        # Validation
        model.eval()
        val_losses = []
        all_targets, all_preds = [], []
        with torch.no_grad():
            for batch in val_loader:
                #print(f"  [DEBUG Val Loop] Processing batch: {batch}")
                batch = batch.to(DEVICE)
                out = model(batch.x, batch.edge_index, batch.edge_attr)
                # Calculate validation loss per item using sigmoid_focal_loss
                # Handle cases where output might be empty if input batch was skipped implicitly by dataloader?
                # Check if batch has targets and edges before calculating loss
                if batch.y.numel() > 0 and batch.edge_index.numel() > 0:
                    loss_per_item = sigmoid_focal_loss(out, batch.y.float(), gamma=GAMMA, reduction='none')
                    # Check if loss calculation itself resulted in empty tensor (shouldn't happen here)
                    if loss_per_item.numel() > 0:
                         val_losses.append(loss_per_item.mean().item()) # Append mean loss for this batch
                    # else: # If loss is empty, append 0.0? Or skip? Let's skip appending if loss is empty.
                    #     val_losses.append(0.0) # Append 0 if loss calculation failed?
                # else: # If batch had no targets/edges, append 0.0 loss for this batch
                #     val_losses.append(0.0)
                # Simplified: Only append loss if batch is valid and loss is calculated
                # This relies on avg_val_loss = np.mean(val_losses) handling empty val_losses if needed
                # Let's ensure val_losses is not empty before np.mean
                #print(f"  [DEBUG Val Loop] batch.y shape: {batch.y.shape}, dtype: {batch.y.dtype}")
                #print(f"  [DEBUG Val Loop] batch.y content (first 10): {batch.y.cpu().numpy()[:10]}")
                all_targets.append(batch.y.cpu().numpy())
                all_preds.append(out.cpu().numpy())
        # Calculate average validation loss, handle case where val_losses might be empty
        avg_val_loss = np.mean(val_losses) if val_losses else 0.0
        y_true = np.concatenate(all_targets)
        y_score = np.concatenate(all_preds)

        # Handle cases where validation set might yield no targets (e.g., empty graphs)
        if y_true.size == 0:
            print(f"  [Warning] Epoch {epoch}: No targets found in validation set. Setting metrics to 0.")
            val_auc = 0.0
            val_f1 = 0.0
            val_recall = 0.0
            val_precision = 0.0
            val_mrr = 0.0
            val_hits1 = 0.0
            val_hits3 = 0.0
            val_hits5 = 0.0
        else:
            val_auc = compute_auc_pr(y_true, y_score)
            val_f1 = f1_score(y_true, y_score > 0.5, zero_division=0)
            val_recall = recall_score(y_true, y_score > 0.5, zero_division=0)
            val_precision = precision_score(y_true, y_score > 0.5, zero_division=0)
            val_mrr = compute_mrr(y_true, y_score)
            val_hits1 = compute_hits_at_k(y_true, y_score, k=1)
            val_hits3 = compute_hits_at_k(y_true, y_score, k=3)
            val_hits5 = compute_hits_at_k(y_true, y_score, k=5)

        # Logging
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("AUC_PR/val", val_auc, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)
        writer.add_scalar("Recall/val", val_recall, epoch)
        writer.add_scalar("Precision/val", val_precision, epoch)
        writer.add_scalar("MRR/val", val_mrr, epoch)
        writer.add_scalar("Hits@1/val", val_hits1, epoch)
        writer.add_scalar("Hits@3/val", val_hits3, epoch)
        writer.add_scalar("Hits@5/val", val_hits5, epoch)

        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | Val AUC-PR={val_auc:.4f}")

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_auc': val_auc,
            }, paths.checkpoint_path(experiment_id=experiment_id)) # Pass experiment_id
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break
        # --- Save validation predictions after training ---
        from scipy.special import expit
        # Save validation predictions inside the experiment's directory
        exp_dir = paths.experiment_dir(experiment_id) # Get experiment dir path
        val_pred_save_path = os.path.join(exp_dir, f"val_pred_scores.npz") # Filename within exp_dir
        np.savez(val_pred_save_path, y_true=y_true, y_score=expit(y_score))
        print(f"[INFO] Saved validation predictions to {val_pred_save_path}")
        scheduler.step(val_auc)
    # No need to return best_val_auc since it is not used


def train_pipeline():
    """
    Standard training pipeline: train/val/test split, no KFold.
    """
    from src.config import save_config_snapshot
    from src import config
    # --- Experiment ID assignment ---
    # In the future, allow passing experiment_id as argument; for now, always increment
    experiment_id = increment_experiment_id()
    print(f"[Experiment Management] Assigned experiment_id: {experiment_id}")
    set_seed(SEED)
    # Create the specific directory for this experiment run
    exp_dir = paths.experiment_dir(experiment_id)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"[Experiment Management] Created experiment directory: {exp_dir}")

    save_config_snapshot(exp_dir) # Save config snapshot inside the experiment dir
    writer = SummaryWriter(exp_dir) # Log TensorBoard data inside the experiment dir
    dataset = WDNLeakDataset(root=DATA_DIR, processed_dir='processed', file_pattern='scenario_*.pt')
    print(f"Full dataset size: {len(dataset)}")

    # Load split indices
    train_idx = np.load(paths.split_index_path("train"))
    val_idx = np.load(paths.split_index_path("val"))
    print(f"Loaded {len(train_idx)} train indices.")

    print(f"Loaded {len(val_idx)} validation indices.")
    if len(val_idx) > 0:
        print(f"Max validation index: {np.max(val_idx)}")
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    print(f"Train subset size: {len(train_set)}")
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Validation subset size: {len(val_set)}")
    # Check if validation set is empty BEFORE creating loader
    if len(val_set) == 0:
        raise ValueError("Validation set is empty. Check dataset path and split indices.")
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    sample = dataset[0]
    node_in_dim = sample.x.shape[1]
    edge_in_dim = sample.edge_attr.shape[1]
    model = WDNLeakGNN(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        gnn_type=GNN_TYPE,
        mlp_hidden_dim=MLP_HIDDEN_DIM,
        dropout=DROPOUT,
        norm_type=NORM_TYPE,
    ).to(DEVICE)

    # criterion = FocalLoss(gamma=GAMMA) # Removed instantiation
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    print("Starting training...")
    # Removed criterion argument from call
    train(model, train_loader, val_loader, optimizer, scheduler, writer, experiment_id) # Pass experiment_id

    # Load best model
    checkpoint = torch.load(paths.checkpoint_path(experiment_id=experiment_id), map_location=DEVICE, weights_only=False) # Pass experiment_id, allow loading non-weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test set evaluation is handled in src/evaluate.py

    writer.close()

if __name__ == "__main__":
    train_pipeline()
