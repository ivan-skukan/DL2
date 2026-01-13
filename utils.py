import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def sample_k_shots(features, labels, k, seed):
    """
    Samples up to k shots per class. If a class has fewer than k samples,
    it takes all available samples for that class.
    """
    torch.manual_seed(seed)
    idxs = []

    for c in torch.unique(labels):
        if c < 0:
            continue
        class_idxs = (labels == c).nonzero(as_tuple=True)[0]
        # Robust sampling: take min(k, available)
        num_to_sample = min(k, len(class_idxs))
        chosen = class_idxs[torch.randperm(len(class_idxs))[:num_to_sample]]
        idxs.append(chosen)

    idxs = torch.cat(idxs)
    return features[idxs], labels[idxs]

def accuracy(logits, labels):
    """Calculates accuracy on the device."""
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()

def ood_metrics(id_scores, ood_scores):
    """
    Calculates AUROC and FPR@95% TPR.
    Scores should be 'confidence' scores (higher = more likely ID).
    """
    # Move to CPU for sklearn metrics
    if torch.is_tensor(id_scores): id_scores = id_scores.cpu().numpy()
    if torch.is_tensor(ood_scores): ood_scores = ood_scores.cpu().numpy()

    y_true = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    y_score = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(y_true, y_score)

    # FPR @ 95% TPR (Threshold at the 5th percentile of ID scores)
    thresh = np.percentile(id_scores, 5)
    fpr95 = (ood_scores >= thresh).mean()

    return float(auroc), float(fpr95)