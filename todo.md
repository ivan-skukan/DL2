# PROJECT TO-DO: Heads Experiments & OOD Evaluation (Revised)

This revised plan integrates the updated code structure, including the `uv` environment, robust sampling, and advanced calibration logic.

## 1. Setup & Sanity Checks

* **Environment**: Confirm the `uv` virtual environment is active and dependencies from `pyproject.toml` are synced.
* **Data Diagnostic**: Run `python check_dataset.py` to verify that local ImageNet-Val and Hugging Face ImageNet-O datasets load correctly.
* **Feature Verification**: Ensure the following files exist in `cached_features/` with 512-dimensional CLIP features:
* `val_features.pt` (ID features and labels).
* `ood_features.pt` (OOD features).
* `text_features.pt` (Class-name embeddings).


* **Shape Check**: Verify tensors match expected shapes: `X_id` , `y_id` , `X_ood` , and `text_features` .

## 2. Experimental Framework

* **Splits**: Use the 80/20 train/test split of In-Distribution (ID) features for few-shot sampling and final evaluation.
* **Robust Sampling**: Use `sample_k_shots` to handle cases where a class might have fewer than  samples by taking all available images for that class.
* **Iteration**: Run experiments for  across seeds .

## 3. Classification Heads Evaluation

Evaluate **ID Accuracy**, **OOD AUROC**, and **FPR@95% TPR** for each configuration.

* **Zero-Shot Head ()**: Use `ZeroShotHead` with text embeddings and maximize cosine similarity.
* **Prototype Head**: Compute class means from -shot samples; ensure features are normalized before and after mean computation.
* **Linear Probe**: Train with Cross-Entropy and Adam, applying weight decay for regularization in low-shot settings.
* **Gaussian Head**: Fit per-class Gaussians with tied covariance and Ledoit-Wolf shrinkage to stabilize Mahalanobis distance scores.

## 4. Calibration & Metrics

* **Temperature Tuning**: Optimize temperature  using `tune_temperature` on the ID test split to minimize Negative Log Likelihood.
* **OOD Thresholding**: Select threshold  using `choose_ood_threshold` at the 5th percentile of ID confidence scores (aiming for 95% TPR).
* **Final Metrics**: Record AUROC and FPR@95% using the unified `ood_metrics` function.

## 5. Visualization

* **Results Aggregation**: Save all experiment metadata and metrics into `results.pt`.
* **Plotting**: Execute `python plot_results.py` to generate side-by-side comparisons of Accuracy and OOD performance across all heads.

---

# Optional: Original Project Path

*The following items represent the original project scope for reference or additional exploration.*

* **Optional Sanity Checks**: Manually visualize image-feature pairs to confirm data integrity.
* **Calibration Variation**: Perform temperature scaling specifically on a small, isolated ID validation subset rather than the full test split.
* **Manual Thresholding**: Experiment with different TPR targets for setting the OOD threshold .
* **Expanded Metrics**:
* Calculate Expected Calibration Error (ECE) and generate reliability plots.
* Measure "Retained ID Accuracy" at specific OOD thresholds.


* **Ablation Studies**:
* Add **SigLIP** as an alternative backbone to compare against CLIP ViT-B/16.
* Test different covariance types (e.g., diagonal vs. full) in the Gaussian head.