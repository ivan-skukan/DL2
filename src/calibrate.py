import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

def temperature_scale(logits, T):
    """Scales logits by temperature. Note: T is used as a divisor."""
    return logits / T

def tune_temperature(logits, labels, name="Head", device="cpu"):
    """
    Optimizes the temperature T using the Cross Entropy Loss on validation data.
    Ensures that labels match logit indices.
    """
    # 1. CRITICAL: Ensure labels are valid for the provided logits
    # This prevents the empty-set optimization bug
    mask = (labels >= 0) & (labels < logits.shape[1])
    safe_logits = logits[mask].to(device)
    safe_labels = labels[mask].to(device)

    if len(safe_labels) == 0:
        print(f"[{name}] WARNING: No valid labels found for calibration. Check logit/label alignment.")
        return 1.0, []

    # 2. Initialization
    # We optimize log_T to ensure T stays positive and to make the landscape smoother
    log_T = nn.Parameter(torch.zeros(1, device=device)) 
    
    # Using LBFGS (standard for temperature scaling)
    optimizer = optim.LBFGS([log_T], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss()

    history = []
    def closure():
        optimizer.zero_grad()
        T = torch.exp(log_T) # Constrain T > 0
        loss = criterion(safe_logits / T, safe_labels)
        loss.backward()
        history.append((T.item(), loss.item()))
        return loss

    print(f"[{name}] Optimizing temperature on {len(safe_labels)} samples...")
    optimizer.step(closure)
    
    final_T = torch.exp(log_T).item()
    print(f"[{name}] Optimization complete. Final T: {final_T:.4f}")
    return final_T, history

def expected_calibration_error(logits, labels, T=1.0, n_bins=15):
    """Calculates the Expected Calibration Error (ECE) with improved binning."""
    scaled_logits = logits / T
    probs = torch.softmax(scaled_logits, dim=1)
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=logits.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i+1]
        # Use logical AND for bin selection
        in_bin = (confidences > bin_lower.item()) & (confidences <= bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()

def plot_reliability_diagram(logits, labels, T=1.0, n_bins=10, name="Head", save_dir="plots"):
    """Enhanced reliability diagram with STER-style visuals."""
    os.makedirs(save_dir, exist_ok=True)
    from .visualize import _finish_style # Use your new styling

    scaled_logits = logits / T
    probs = torch.softmax(scaled_logits, dim=1)
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if in_bin.any():
            bin_accs.append(accuracies[in_bin].float().mean().item())
            bin_confs.append(confidences[in_bin].mean().item())
        else:
            bin_accs.append(0)
            bin_confs.append(bin_boundaries[i].item() + 0.05)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    
    # Bars represent accuracy per bin
    ax.bar(bin_boundaries[:-1].numpy(), bin_accs, width=1/n_bins, align='edge', 
           alpha=0.6, color='#d17a22', edgecolor='#2a1f1c', label='Accuracy', zorder=3)
    
    # Diagonal represents perfect calibration
    ax.plot([0, 1], [0, 1], '--', color='#6b705c', lw=2, label='Perfectly Calibrated', zorder=4)
    
    ax.legend(frameon=False)
    _finish_style(ax, f"Reliability: {name} (T={T:.2f})", "Confidence", "Accuracy")
    fig.savefig(f"{save_dir}/{name}_reliability.png")
    plt.close()