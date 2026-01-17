import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from sklearn.manifold import TSNE
import os

def plot_confidence_histograms(id_conf, ood_conf, name="Head", save_dir="plots"):
    """Plots overlapping histograms of ID and OOD confidence scores."""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(id_conf.cpu().numpy(), bins=50, alpha=0.5, label='In-Distribution (ID)', color='blue', density=True)
    plt.hist(ood_conf.cpu().numpy(), bins=50, alpha=0.5, label='Out-of-Distribution (OOD)', color='red', density=True)
    plt.xlabel('Maximum Softmax Confidence')
    plt.ylabel('Density')
    plt.title(f'Confidence Distribution: {name}')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"{save_dir}/{name}_conf_hist.png")
    plt.close()

def plot_pr_curve(id_scores, ood_scores, name="Head", save_dir="plots"):
    """Plots the Precision-Recall curve for OOD detection."""
    os.makedirs(save_dir, exist_ok=True)
    # ID is positive class (1), OOD is negative class (0)
    scores = np.concatenate([id_scores.cpu().numpy(), ood_scores.cpu().numpy()])
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(7, 7))
    plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.4f})', color='darkorange', lw=2)
    plt.xlabel('Recall (TPR)')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve: {name}')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(f"{save_dir}/{name}_pr_curve.png")
    plt.close()

def plot_feature_embeddings(X_id, X_ood, y_id, name="Head", n_samples=500, save_dir="plots"):
    """Visualizes ID and OOD features in 2D using t-SNE."""
    os.makedirs(save_dir, exist_ok=True)
    idx_id = np.random.choice(len(X_id), min(n_samples, len(X_id)), replace=False)
    idx_ood = np.random.choice(len(X_ood), min(n_samples, len(X_ood)), replace=False)
    
    features = torch.cat([X_id[idx_id], X_ood[idx_ood]]).cpu().numpy()
    labels = y_id[idx_id].cpu().numpy()
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 7))
    # Plot ID samples colored by class
    plt.scatter(embeddings[:len(idx_id), 0], embeddings[:len(idx_id), 1], 
                c=labels, cmap='tab20', s=15, alpha=0.8)
    
    # Plot OOD samples in gray
    plt.scatter(embeddings[len(idx_id):, 0], embeddings[len(idx_id):, 1], 
                c='gray', s=10, alpha=0.3, marker='x')
    
    plt.title(f't-SNE Visualization: {name}')
    
    # FIXED LEGEND: Create two proxy handles for a clean legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='ID Classes', 
               markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='x', color='gray', label='OOD Samples', 
               linestyle='None', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.savefig(f"{save_dir}/{name}_tsne.png")
    plt.close()

def plot_retained_acc_vs_rejection(id_conf, id_labels, preds_id, name="Head", save_dir="plots"):
    """Plots ID Accuracy vs. Rejection Rate as the OOD threshold is varied."""
    os.makedirs(save_dir, exist_ok=True)
    thresholds = np.linspace(0, 1, 100)
    accuracies = []
    rejection_rates = []
    
    for t in thresholds:
        mask = id_conf >= t
        if mask.sum() > 0:
            acc = (preds_id[mask] == id_labels[mask]).float().mean().item()
            accuracies.append(acc)
            rejection_rates.append(1.0 - (mask.sum().item() / len(id_conf)))
        else:
            break

    plt.figure(figsize=(8, 5))
    plt.plot(rejection_rates, accuracies, marker='.', color='green')
    plt.xlabel('Proportion of ID Data Rejected')
    plt.ylabel('Accuracy on Remaining ID Data')
    plt.title(f'Retained Accuracy vs. Rejection Rate: {name}')
    plt.grid(True)
    plt.savefig(f"{save_dir}/{name}_retained_acc.png")
    plt.close()