# CLIP Analysis: OOD Detection & Calibration

This project analyzes CLIP metrics across different classification heads, specifically focusing on the trade-off between In-Distribution (ID) accuracy and Out-of-Distribution (OOD) detection performance.

**Datasets:** - **ImageNet-1k**: Used for In-Distribution (ID) evaluation.

* **ImageNet-O**: Used for Out-of-Distribution (OOD) evaluation.

**Backbone:** ViT-B/16.

## 1. Project Components

The framework evaluates four primary classification heads:

* **Zero-Shot Head**: Uses text embeddings to define class centroids via cosine similarity.
* **Prototype Head**: Computes class means from -shot samples.
* **Linear Probe**: A supervised linear layer trained to map CLIP features to ID labels.
* **Gaussian Head**: Models class distributions using Mahalanobis distance with tied covariance and shrinkage.

## 2. Setup

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

1. **Datasets**: Ensure **ImageNet-1k** (Val) is local and **ImageNet-O** is accessible via Hugging Face.
2. **Embeddings**: Generate or place 512-dimensional CLIP features in `cached_features/`:
* `val_features.pt` (ID)
* `ood_features.pt` (OOD)
* `text_features.pt` (Class-name embeddings)



## 3. Running the Pipeline

You can run the entire experiment, visualization, and summary process using the provided automation script:

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh

```

### Manual Steps

1. **Sanity Check**: `python check_dataset.py` verifies data loading.
2. **Feature Extraction**: Run `embed.py` and `text_embed.py` if features are missing.
3. **Main Experiment**: `python main.py` evaluates all heads across -shots and seeds, generating `results.pt`.
4. **Results Summary**: `python src/summary.py` generates a human-readable Markdown table of averaged results.
5. **Visualization**: `python plot_results.py` generates standard performance plots.

## 4. Key Features

### Calibration & Metrics (Task 4)

The project implements advanced calibration logic to ensure model confidence reflects actual accuracy:

* **Temperature Tuning**: Optimizes a scalar  using LBFGS to minimize Negative Log Likelihood (NLL).
* **Expected Calibration Error (ECE)**: Quantifies the discrepancy between accuracy and confidence.
* **Unified OOD Metrics**: Calculates **AUROC** and **FPR@95% TPR** (setting thresholds at the 5th percentile of ID confidence).

### Advanced Visualizations

Generated in the `plots/` directory for each head:

* **Reliability Diagrams**: Visualizing calibration quality pre- and post-tuning.
* **Confidence Histograms**: Overlapping ID vs. OOD score distributions.
* **Precision-Recall Curves**: Secondary OOD performance metric.
* **t-SNE Embeddings**: 2D visualization of the CLIP feature space.
* **Retained Accuracy**: Plotting ID accuracy vs. data rejection rates.

## 5. Experimental Insights

* **Linear Probe**: Achieves the highest ID accuracy but is prone to significant miscalibration (high ECE).
* **Prototype Head**: Offers the most stable OOD detection performance (AUROC) as  increases and remains well-calibrated.
* **Gaussian Head**: Demonstrates instability in high-dimensional CLIP spaces with tied covariance, particularly in very low-shot () scenarios.