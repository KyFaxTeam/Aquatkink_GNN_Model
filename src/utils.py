import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve, auc

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_auc_pr(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)

def compute_mrr(y_true, y_score):
    # y_true: (num_edges,) binary, y_score: (num_edges,) float
    # Only one positive per sample (leak localization)
    order = np.argsort(-y_score)
    ranks = np.where(y_true[order] == 1)[0]
    if len(ranks) == 0:
        return 0.0
    return 1.0 / (ranks[0] + 1)

def compute_hits_at_k(y_true, y_score, k=1):
    order = np.argsort(-y_score)
    top_k = order[:k]
    return int(np.any(y_true[top_k] == 1))

# Custom FocalLoss class removed, will use torchvision.ops.sigmoid_focal_loss instead