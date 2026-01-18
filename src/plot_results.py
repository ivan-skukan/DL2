import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from visualize import _finish_style, OPTS, WARM_PALETTE

def plot_metrics(results_path):
    data = torch.load(results_path, weights_only=False) 
    df = pd.DataFrame(data)
    metrics = ['acc', 'auroc', 'fpr95', 'ece']
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), dpi=150)
    sns.set_palette(WARM_PALETTE)

    for i, metric in enumerate(metrics):
        plot_df = df.dropna(subset=[metric])
        sns.lineplot(data=plot_df, x='K', y=metric, hue='head', marker='o', 
                     linewidth=2.5, markersize=10, ax=axes[i])
        
        axes[i].legend(frameon=False, loc='best')
        _finish_style(axes[i], f'{metric.upper()} vs K-Shots', "K (Shots)", metric.upper())

    plt.tight_layout()
    plt.savefig('results_plot.png')
    print("Main line plots saved as results_plot.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results.pt")
    args = parser.parse_args()
    plot_metrics(args.results)