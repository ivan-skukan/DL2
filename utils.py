import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def sample_k_shots(features, labels, k, seed):
    torch.manual_seed(seed)
    idxs = []

    for c in torch.unique(labels):
        if c < 0:
            continue
        class_idxs = (labels == c).nonzero(as_tuple=True)[0]
        if len(class_idxs) >= k:
            chosen = class_idxs[torch.randperm(len(class_idxs))[:k]]
            idxs.append(chosen)

    idxs = torch.cat(idxs)
    return features[idxs], labels[idxs]

def accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()

def ood_metrics(id_scores, ood_scores):
    y_true = np.concatenate([
        np.ones(len(id_scores)),
        np.zeros(len(ood_scores))
    ])
    y_score = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(y_true, y_score)

    # FPR @ 95% TPR
    thresh = np.percentile(id_scores, 5)
    fpr95 = (ood_scores >= thresh).mean()

    return auroc, fpr95
