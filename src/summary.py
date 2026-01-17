import torch
import pandas as pd

def generate_summary(results_path="results.pt"):
    # Load results
    data = torch.load(results_path, weights_only=False)
    df = pd.DataFrame(data)

    # Group by Head and K, then average across seeds
    summary = df.groupby(['head', 'K']).agg({
        'acc': 'mean',
        'auroc': 'mean',
        'fpr95': 'mean',
        'ece': 'mean'
    }).reset_index()

    # Format for readability
    print("### Project Experiment Summary (Averaged over Seeds)")
    print("| Head | K-Shots | Accuracy (%) | AUROC | FPR@95% | ECE |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for _, row in summary.sort_values(['K', 'auroc'], ascending=[True, False]).iterrows():
        print(f"| {row['head']} | {int(row['K'])} | {row['acc']*100:.2f}% | {row['auroc']:.4f} | {row['fpr95']:.4f} | {row['ece']:.4f} |")

if __name__ == "__main__":
    generate_summary()