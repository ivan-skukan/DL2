# CLIP Analysis: OOD Detection & Calibration

This project analyzes CLIP metrics across different classification heads, specifically focusing on the trade-off between In-Distribution (ID) accuracy and Out-of-Distribution (OOD) detection performance.

**Backbone:** ViT-B/16.

**Datasets:** ImageNet-1k (ID) and ImageNet-O (OOD).

## Key Findings

* 
**Gaussian Head**: While it achieves the highest ID accuracy at  (68.9%), it demonstrates extreme instability in low-shot scenarios, achieving only 0.11% accuracy due to the curse of dimensionality.


* 
**Prototype Head**: This head provides the most stable and effective OOD detection performance, reaching a peak AUROC of 0.782 at  by preserving the semantic structure of CLIP.


* 
**Linear Probe**: This approach is prone to significant miscalibration and poor OOD detection because it optimizes for specific classification boundaries rather than the underlying distribution density.


* 
**Zero-Shot Baseline**: The zero-shot CLIP model remains highly competitive, outperforming all tested few-shot methods when provided with fewer than 8 shots.


## Features

### Calibration & Metrics

The project implements calibration logic to ensure model confidence reflects actual accuracy:

* 
**Temperature Tuning**: Optimizes a scalar using LBFGS to minimize Negative Log Likelihood (NLL).


* 
**Expected Calibration Error (ECE)**: Quantifies the discrepancy between accuracy and confidence.


* 
**Unified OOD Metrics**: Calculates **AUROC** and **FPR@95% TPR**.



### Visualizations

The following are generated in the `plots/` directory for each head:

* 
**Reliability Diagrams**: Visualizing calibration quality pre- and post-tuning.


* 
**Confidence Histograms**: Overlapping ID vs. OOD score distributions.


* 
**Precision-Recall Curves**: Secondary OOD performance metric.


* 
**t-SNE Embeddings**: 2D visualization of the CLIP feature space.


* 
**Retained Accuracy**: Plotting ID accuracy vs. data rejection rates.


## Setup

### Environment Setup with `uv`

This project uses `uv` for package management.

```bash
# Create and activate environment
uv venv
source .venv/bin/activate

# Install dependencies
uv sync

```

### Data & Features

1. 
**Datasets**: Ensure **ImageNet-1k** (Val) is local and **ImageNet-O** is accessible via Hugging Face.


2. 
**Embeddings**: Place 512-dimensional CLIP features in `cached_features/`:


* `val_features.pt` (ID)
* `ood_features.pt` (OOD)
* `text_features.pt` (Class-name embeddings)



## 3. Running the Pipeline

Run the entire experiment, visualization, and summary process using the automation script:

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh

```

### Manual Steps

1. 
**Sanity Check**: `python check_dataset.py` verifies data loading.


2. 
**Feature Extraction**: Run `embed.py` and `text_embed.py` if features are missing.


3. 
**Main Experiment**: `python main.py` evaluates all heads across -shots and seeds, generating `results.pt`.


4. 
**Results Summary**: `python src/summary.py` generates a human-readable Markdown table of averaged results.


5. 
**Visualization**: `python plot_results.py` generates standard performance plots.
