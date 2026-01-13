import torch
import torch.optim as optim
import torch.nn as nn

def temperature_scale(logits, T):
    """Applies temperature scaling to logits."""
    return logits / T

def tune_temperature(logits, labels, device="cpu"):
    """
    Finds the optimal temperature T by minimizing CrossEntropyLoss on a validation set.
    """
    T = nn.Parameter(torch.ones(1, device=device))
    optimizer = optim.LBFGS([T], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss()

    def eval():
        optimizer.zero_grad()
        loss = criterion(temperature_scale(logits, T), labels)
        loss.backward()
        return loss

    optimizer.step(eval)
    return T.item()

def choose_ood_threshold(val_conf, percentile=5):
    """
    Chooses a threshold tau such that 'percentile'% of ID samples are below it.
    Used for setting FPR@95% TPR.
    """
    tau = torch.quantile(val_conf, percentile / 100)
    return tau