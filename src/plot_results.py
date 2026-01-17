import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def plot_metrics(results_path):
    data = torch.load(results_path, weights_only=False) 
    df = pd.DataFrame(data)

    # ADD 'ece' to this list
    metrics = ['acc', 'auroc', 'fpr95', 'ece']
    fig, axes = plt.subplots(1, 4, figsize=(24, 5)) # Increased size for 4 plots

    for i, metric in enumerate(metrics):
        plot_df = df.dropna(subset=[metric])
        sns.lineplot(data=plot_df, x='K', y=metric, hue='head', marker='o', ax=axes[i])
        axes[i].set_title(f'{metric.upper()} vs K-Shots')
        axes[i].grid(True)

    plt.tight_layout()
    plt.savefig('results_plot.png')
    print("Plot updated with ECE and saved as results_plot.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results.pt")
    args = parser.parse_args()
    plot_metrics(args.results)