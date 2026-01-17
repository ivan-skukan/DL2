import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load results
results = torch.load('results.pt')
df = pd.DataFrame(results)

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Few-Shot OOD Detection Performance Summary', fontsize=16, fontweight='bold')

# Prepare data
methods = ['ZeroShot', 'Prototype', 'LinearProbe', 'Gaussian']
k_values = [0, 1, 2, 4, 8, 16]

# ============================================================================
# Plot 1: Accuracy by K
# ============================================================================
ax = axes[0, 0]
for method in methods:
    if method == 'ZeroShot':
        method_df = df[df['head'] == method]
        acc_means = [method_df['acc'].values[0]]
        k_plot = [0]
    else:
        method_df = df[df['head'] == method]
        acc_means = [method_df[method_df['K']==k]['acc'].mean() for k in [1,2,4,8,16]]
        k_plot = [1,2,4,8,16]
    
    ax.plot(k_plot, acc_means, marker='o', linewidth=2, markersize=8, label=method)

ax.set_xlabel('K (Number of Shots)', fontsize=11, fontweight='bold')
ax.set_ylabel('In-Distribution Accuracy', fontsize=11, fontweight='bold')
ax.set_title('(A) ID Accuracy vs K-Shot', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 0.75])

# ============================================================================
# Plot 2: AUROC by K
# ============================================================================
ax = axes[0, 1]
for method in methods:
    if method == 'ZeroShot':
        method_df = df[df['head'] == method]
        auroc_means = [method_df['auroc'].values[0]]
        k_plot = [0]
    else:
        method_df = df[df['head'] == method]
        auroc_means = [method_df[method_df['K']==k]['auroc'].mean() for k in [1,2,4,8,16]]
        k_plot = [1,2,4,8,16]
    
    ax.plot(k_plot, auroc_means, marker='s', linewidth=2, markersize=8, label=method)

ax.set_xlabel('K (Number of Shots)', fontsize=11, fontweight='bold')
ax.set_ylabel('OOD Detection AUROC', fontsize=11, fontweight='bold')
ax.set_title('(B) OOD AUROC vs K-Shot', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.45, 0.80])
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')

# ============================================================================
# Plot 3: FPR@95 by K (lower is better)
# ============================================================================
ax = axes[1, 0]
for method in methods:
    if method == 'ZeroShot':
        method_df = df[df['head'] == method]
        fpr_means = [method_df['fpr95'].values[0]]
        k_plot = [0]
    else:
        method_df = df[df['head'] == method]
        fpr_means = [method_df[method_df['K']==k]['fpr95'].mean() for k in [1,2,4,8,16]]
        k_plot = [1,2,4,8,16]
    
    ax.plot(k_plot, fpr_means, marker='^', linewidth=2, markersize=8, label=method)

ax.set_xlabel('K (Number of Shots)', fontsize=11, fontweight='bold')
ax.set_ylabel('FPR@95 (lower is better)', fontsize=11, fontweight='bold')
ax.set_title('(C) False Positive Rate at 95% TPR', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.75, 1.05])
ax.invert_yaxis()  # Invert so lower is visually better

# ============================================================================
# Plot 4: Accuracy-AUROC Scatter (K=16 only)
# ============================================================================
ax = axes[1, 1]
k16_df = df[df['K'].isin([0, 16])]

for method in methods:
    if method == 'ZeroShot':
        method_df = k16_df[(k16_df['head'] == method) & (k16_df['K'] == 0)]
    else:
        method_df = k16_df[(k16_df['head'] == method) & (k16_df['K'] == 16)]
    
    if len(method_df) > 0:
        acc = method_df['acc'].mean()
        auroc = method_df['auroc'].mean()
        ax.scatter(acc, auroc, s=300, marker='o', alpha=0.7, label=method)
        ax.annotate(method, (acc, auroc), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold')

ax.set_xlabel('In-Distribution Accuracy', fontsize=11, fontweight='bold')
ax.set_ylabel('OOD Detection AUROC', fontsize=11, fontweight='bold')
ax.set_title('(D) Accuracy vs AUROC Trade-off (K=0 or K=16)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 0.75])
ax.set_ylim([0.45, 0.80])

# Add reference lines
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.3)

# Add optimal region annotation
ax.fill_between([0.5, 0.75], [0.75, 0.75], [0.8, 0.8], alpha=0.1, color='green', label='Optimal Region')

plt.tight_layout()
plt.savefig('plots/comprehensive_summary.png', dpi=300, bbox_inches='tight')
print("Saved comprehensive summary plot to plots/comprehensive_summary.png")

# ============================================================================
# Create a summary table image
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Prepare table data
table_data = []
table_data.append(['K', 'Method', 'Accuracy', 'AUROC', 'FPR@95', 'ECE'])

# Add zero-shot
zs = df[df['K'] == 0].iloc[0]
table_data.append(['0', 'ZeroShot', f"{zs['acc']:.3f}", f"{zs['auroc']:.3f}", 
                  f"{zs['fpr95']:.3f}", f"{zs['ece']:.3f}"])

# Add K=1,8,16 for each method
for k in [1, 8, 16]:
    for method in ['Prototype', 'LinearProbe', 'Gaussian']:
        method_df = df[(df['K'] == k) & (df['head'] == method)]
        if len(method_df) > 0:
            acc = method_df['acc'].mean()
            auroc = method_df['auroc'].mean()
            fpr = method_df['fpr95'].mean()
            ece = method_df['ece'].mean()
            table_data.append([str(k), method, f"{acc:.3f}", f"{auroc:.3f}", 
                             f"{fpr:.3f}", f"{ece:.3f}"])

# Create table
table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                colWidths=[0.08, 0.18, 0.15, 0.15, 0.15, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header row
for i in range(6):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight best values
# Find best for each K
for row in range(1, len(table_data)):
    if table_data[row][0] != '0':  # Skip zero-shot
        table[(row, 0)].set_facecolor('#E8F5E9')

# Add title
ax.text(0.5, 0.95, 'Performance Summary Table', 
        horizontalalignment='center', fontsize=14, fontweight='bold',
        transform=ax.transAxes)

plt.savefig('plots/summary_table.png', dpi=300, bbox_inches='tight')
print("Saved summary table to plots/summary_table.png")

print("\n✓ All presentation visuals generated!")
