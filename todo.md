Based on the progress made in implementing the experimental pipeline, calibration logic, and advanced visualizations, here is the updated `todo.md` focusing on the final stages of analysis and presentation.

# PROJECT TO-DO: Analysis & Presentation

## 1. Result Analysis & Interpretation

* **Deep Dive into Gaussian Failure**: Document the specific reasons for the Gaussian head's failure at  and its stagnation at random-chance levels, focusing on the "curse of dimensionality" and the limitations of tied-covariance in high-dimensional CLIP space.
* **Accuracy vs. OOD Trade-off**: Finalize the interpretation of why the Linear Probe excels in ID Accuracy while the Prototype head dominates in OOD AUROC and FPR@95%.
* **Calibration Verification**: Summarize the impact of temperature scaling on Expected Calibration Error (ECE) and verify that calibrated scores provide more reliable OOD thresholds.

## 2. Asset Finalization

* **Organize Visualizations**: Audit the `plots/` directory to ensure all "hero" visuals (Confidence Histograms, PR Curves, t-SNE, and Reliability Diagrams) are generated for the best-performing -shot scenarios (typically ).
* **Final Summary Table**: Use `summary.py` to generate the final averaged results table for inclusion in the presentation.

## 3. Presentation Preparation

* **Slide Construction**: Build the presentation deck following the established outline:
* **Methodology**: CLIP ViT-B/16 and the four classification heads.
* **Theory**: The mechanics of Temperature Scaling and ECE.
* **The "Paradox"**: Explain the discrepancy between low classification accuracy and high OOD detection performance.
* **Decision Logic**: Present the "Retained Accuracy vs. Rejection" curves as a practical deployment metric.



## 4. (Optional) Targeted Ablations

* **SigLIP Comparison**: Compare CLIP results against a SigLIP backbone to determine if alternative pre-training objectives improve OOD robustness.
* **Covariance Refinement**: Briefly test if a diagonal covariance matrix (rather than tied-full) stabilizes the Gaussian head's performance in low-shot settings.