import torch
import torch.nn.functional as F

def temperature_scale(logits, T):
    return logits / T

def choose_ood_threshold(val_conf, val_is_id, percentile=5):
    # e.g., threshold at 5th percentile of ID confidence
    id_conf = val_conf[val_is_id]
    tau = torch.quantile(id_conf, percentile/100)
    return tau
