import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.metrics import precision_recall_curve, auc
from sklearn.manifold import TSNE
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D

# =========================
# STER-Style Global Options
# =========================
# Values derived from ster/src/analysis/visualize_runs.py
OPTS = {
    "show_values": True,
    "value_fmt": "{:.2f}",
    "hide_spines": True,
    "font_title": 12,
    "font_axis": 11,
    "font_tick": 10,
    "font_value": 10,
    "slide_small": True,      # Zoomed-in mode for slide readability
    "slide_scale": 1.35,      # Scaling factor for fonts
    "slide_value_scale": 1.40
}

# Warm, readable palette from ster
WARM_PALETTE = [
    "#f2c9a1", "#e5989b", "#d17a22", "#b56576", "#f6bd60",
    "#b55d4c", "#cd9a7b", "#f28482", "#c7a27e", "#6b705c",
]

def get_color(i):
    return WARM_PALETTE[i % len(WARM_PALETTE)]

def _finish_style(ax, title, xlabel, ylabel):
    """Applies consistent ster-repo styling and typography scaling."""
    scale = OPTS["slide_scale"] if OPTS["slide_small"] else 1.0
    
    if title:
        ax.set_title(title, fontsize=int(OPTS["font_title"] * scale), weight='bold', pad=15)
    
    # Border and spine styling
    if OPTS["hide_spines"]:
        for s in ["top", "right"]: 
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color("#7f675f")
            ax.spines[s].set_alpha(0.6)
            ax.spines[s].set_linewidth(0.8)

    ax.set_xlabel(xlabel, fontsize=int(OPTS["font_axis"] * scale))
    ax.set_ylabel(ylabel, fontsize=int(OPTS["font_axis"] * scale))
    
    ax.tick_params(axis='both', which='major', labelsize=int(OPTS["font_tick"] * scale), 
                   color="#7f675f", direction="out", length=4)
    
    # Grid styling
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.55, zorder=0)
    plt.tight_layout()

def _add_shadows(bars):
    """Applies soft shadows to bar patches."""
    for b in bars:
        b.set_path_effects([pe.SimplePatchShadow(offset=(1.2, -1.2), alpha=0.22), pe.Normal()])

# =========================
# General Summary Plot
# =========================
def plot_general_summary_bars(results_path, save_dir="plots"):
    """
    Creates high-impact summary bar charts comparing all heads at the 
    highest K-shot count to show overall results for the presentation.
    """
    os.makedirs(save_dir, exist_ok=True)
    data = torch.load(results_path, weights_only=False)
    df = pd.DataFrame(data)
    
    # Filter for max K-shots to show best-case performance
    max_k = df['K'].max()
    summary_df = df[df['K'] == max_k].groupby('head').mean(numeric_only=True).reset_index()
    
    metrics = [('acc', 'Accuracy'), ('auroc', 'AUROC'), ('fpr95', 'FPR@95%')]
    
    for metric_key, label in metrics:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        heads = summary_df['head'].tolist()
        values = summary_df[metric_key].tolist()
        
        bars = ax.bar(heads, values, color=[get_color(i) for i in range(len(heads))], 
                      edgecolor="#2a1f1c", linewidth=0.9, zorder=3)
        _add_shadows(bars)
        
        # Add value labels with boosted slide font size
        if OPTS["show_values"]:
            v_scale = OPTS["slide_value_scale"] if OPTS["slide_small"] else 1.0
            fs = int(OPTS["font_value"] * v_scale)
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=fs, color="#2a1f1c",
                            path_effects=[pe.withStroke(linewidth=1.4, foreground="white")])

        _finish_style(ax, f"Final Performance Comparison: {label} (K={max_k})", 
                      "Classification Head", label)
        fig.savefig(f"{save_dir}/summary_bars_{metric_key}.png")
        plt.close(fig)

# =========================
# OOD Detection Visuals
# =========================
def plot_confidence_histograms(id_conf, ood_conf, name="Head", save_dir="plots"):
    """Plots overlapping ID vs OOD histograms with zoomed styling."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    
    ax.hist(id_conf.cpu().numpy(), bins=50, alpha=0.6, label='ID (ImageNet-1k)', 
            color='#d17a22', density=True, zorder=2)
    ax.hist(ood_conf.cpu().numpy(), bins=50, alpha=0.6, label='OOD (ImageNet-O)', 
            color='#b56576', density=True, zorder=2)
    
    ax.legend(frameon=False, fontsize=11 * OPTS["slide_scale"])
    _finish_style(ax, f"Confidence Distribution: {name}", "Softmax Confidence", "Density")
    fig.savefig(f"{save_dir}/{name}_conf_hist.png")
    plt.close()

def plot_feature_embeddings(X_id, X_ood, y_id, name="Head", n_samples=500, save_dir="plots"):
    """t-SNE visualization with ster-style points and clean legend."""
    os.makedirs(save_dir, exist_ok=True)
    
    idx_id = np.random.choice(len(X_id), min(n_samples, len(X_id)), replace=False)
    idx_ood = np.random.choice(len(X_ood), min(n_samples, len(X_ood)), replace=False)
    
    features = torch.cat([X_id[idx_id], X_ood[idx_ood]]).cpu().numpy()
    labels = y_id[idx_id].cpu().numpy()
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    
    # ID Classes Scatter
    ax.scatter(embeddings[:len(idx_id), 0], embeddings[:len(idx_id), 1], 
                c=labels, cmap='tab20', s=30, alpha=0.75, edgecolors='white', 
                linewidth=0.5, zorder=3)
    
    # OOD Scatter
    ax.scatter(embeddings[len(idx_id):, 0], embeddings[len(idx_id):, 1], 
                c='gray', s=20, alpha=0.3, marker='x', zorder=2)
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Known ID Classes', 
               markerfacecolor='#d17a22', markersize=10),
        Line2D([0], [0], marker='x', color='gray', label='Unknown OOD Samples', 
               linestyle='None', markersize=10)
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=False, 
              fontsize=10 * OPTS["slide_scale"])
    
    _finish_style(ax, f"CLIP Feature Space (t-SNE): {name}", "Latent Dim 1", "Latent Dim 2")
    fig.savefig(f"{save_dir}/{name}_tsne.png")
    plt.close()

def plot_pr_curve(id_scores, ood_scores, name="Head", save_dir="plots"):
    """Precision-Recall curve with presentation-ready styling."""
    os.makedirs(save_dir, exist_ok=True)
    scores = np.concatenate([id_scores.cpu().numpy(), ood_scores.cpu().numpy()])
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    ax.plot(recall, precision, label=f'PR (AUC = {pr_auc:.4f})', color='#b55d4c', lw=3, zorder=3)
    
    ax.legend(loc="lower left", frameon=False, fontsize=10 * OPTS["slide_scale"])
    _finish_style(ax, f"OOD Precision-Recall: {name}", "Recall (TPR)", "Precision")
    fig.savefig(f"{save_dir}/{name}_pr_curve.png")
    plt.close()

def plot_retained_acc_vs_rejection(id_conf, id_labels, preds_id, name="Head", save_dir="plots"):
    """Plots ID Accuracy vs. Proportion of Data Rejected."""
    os.makedirs(save_dir, exist_ok=True)
    thresholds = np.linspace(0, 1, 100)
    accuracies, rejection_rates = [], []
    
    for t in thresholds:
        mask = id_conf >= t
        if mask.sum() > 0:
            acc = (preds_id[mask] == id_labels[mask]).float().mean().item()
            accuracies.append(acc)
            rejection_rates.append(1.0 - (mask.sum().item() / len(id_conf)))
        else:
            break

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(rejection_rates, accuracies, marker='o', markersize=4, color='#6b705c', lw=2, zorder=3)
    
    _finish_style(ax, f"Accuracy/Rejection Trade-off: {name}", 
                  "Proportion of Data Rejected", "Accuracy on Remaining Data")
    fig.savefig(f"{save_dir}/{name}_retained_acc.png")
    plt.close()