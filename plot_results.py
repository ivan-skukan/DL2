import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def plot_metrics(results_path):
    data = torch.load(results_path)
    df = pd.DataFrame(data)

    metrics = ['acc', 'auroc', 'fpr95']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, metric in enumerate(metrics):
        sns.lineplot(data=df, x='K', y=metric, hue='head', marker='o', ax=axes[i])
        axes[i].set_title(f'{metric.upper()} vs K-Shots')
        axes[i].grid(True)

    plt.tight_layout()
    plt.savefig('results_plot.png')
    print("Plot saved as results_plot.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results.pt")
    args = parser.parse_args()
    plot_metrics(args.results)