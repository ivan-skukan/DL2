import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import os

def temperature_scale(logits, T):
    return logits / T

def tune_temperature(logits, labels, name="Head", device="cpu"):
    # Filter labels to be in range
    mask = (labels >= 0) & (labels < logits.shape[1])
    safe_logits, safe_labels = logits[mask], labels[mask]

    T = nn.Parameter(torch.ones(1, device=device))
    optimizer = optim.LBFGS([T], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss()

    history = []
    def eval():
        optimizer.zero_grad()
        loss = criterion(temperature_scale(safe_logits, T), safe_labels)
        loss.backward()
        history.append((T.item(), loss.item()))
        return loss

    print(f"[{name}] Starting Calibration...")
    optimizer.step(eval)
    return T.item(), history

def expected_calibration_error(logits, labels, T=1.0, n_bins=10):
    """Calculates the Expected Calibration Error (ECE)."""
    scaled_logits = logits / T
    probs = torch.softmax(scaled_logits, dim=1)
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=logits.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i+1]
        in_bin = confidences.gt(bin_lower.item()) & confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()

def plot_reliability_diagram(logits, labels, T=1.0, n_bins=10, name="Head", save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    scaled_logits = logits / T
    probs = torch.softmax(scaled_logits, dim=1)
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_accs = []
    for i in range(n_bins):
        in_bin = confidences.gt(bin_boundaries[i]) & confidences.le(bin_boundaries[i+1])
        if in_bin.any():
            bin_accs.append(accuracies[in_bin].float().mean().item())
        else:
            bin_accs.append(0)

    plt.figure(figsize=(6, 6))
    plt.bar(bin_boundaries[:-1].numpy(), bin_accs, width=1/n_bins, align='edge', alpha=0.4, edgecolor='black')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f'Reliability Diagram: {name} (T={T:.2f})')
    plt.savefig(f"{save_dir}/{name}_reliability.png")
    plt.close()