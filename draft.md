### Understanding the ID Accuracy vs. OOD AUROC Discrepancy

It is common in OOD detection research to see models with low classification accuracy but high OOD detection performance. This happens because the two metrics measure fundamentally different capabilities:

* **ID Accuracy (Classification)**: This measures the model's ability to choose the *correct* class from a specific set of candidates (e.g., distinguishing between 1,000 different breeds of dogs). If the model’s internal representations for "Golden Retriever" and "Labrador" are slightly misaligned, it will fail the classification task, leading to low accuracy.
* **OOD AUROC (Detection)**: This measures the model's ability to distinguish the *entire set* of known classes from unknown data. A model can be completely wrong about whether an image is a "Golden Retriever" or a "Labrador" (zero accuracy) while still being very certain that the image is a "Dog" and *not* a "Toaster" or a "Random Pattern".

In the Zero-Shot and Prototype experiments, the low accuracy suggests a mapping mismatch—the model isn't picking the specific class index correctly. However, the high AUROC shows that the model still assigns significantly higher confidence to *any* familiar class than it does to the alien samples in ImageNet-O.

---

### Revised Presentation Draft: Theory and Results

#### **Slide 1: Title & Overview**

* **Topic**: Robust Few-Shot Out-of-Distribution (OOD) Detection.
* **Context**: Leveraging CLIP's pre-trained vision-language features for reliable classification and uncertainty estimation in the ImageNet domain.

#### **Slide 2: The Core Objective**

* **Goal**: To determine if augmenting pre-trained models with a few labeled samples (few-shot learning) can improve the reliability of OOD detection without sacrificing the model's inherent generalization capabilities.
* **Primary Metrics**: Accuracy for In-Distribution (ID) utility and AUROC/FPR@95% for Out-of-Distribution (OOD) safety.

#### **Slide 3: Theoretical Framework: The Four Heads**

* **Zero-Shot**: A baseline using text-based semantic embeddings to define class centroids.
* **Prototype Head**: Defines classes by their empirical mean in the feature space; optimized for stability as more samples are added.
* **Gaussian Head**: Models each class as a multivariate distribution using Mahalanobis distance to capture the "shape" and "spread" of class clusters.
* **Linear Probe**: A discriminative layer that learns a direct decision boundary between classes through supervised fine-tuning.

#### **Slide 4: Calibration Theory: From Scores to Probabilities**

* **The Problem**: Raw model outputs (logits) are often uncalibrated, meaning their "confidence" doesn't reflect actual likelihood.
* **The Solution**: Temperature Scaling () is used to rescale the logit distribution, minimizing the Expected Calibration Error (ECE).
* **Why it Matters**: Proper calibration is essential for setting reliable OOD thresholds that maintain a consistent True Positive Rate.

#### **Slide 5: Performance Analysis: Accuracy vs. OOD Robustness**

* **Classification Winner**: The Linear Probe achieves superior accuracy (up to 52%) by learning specific data patterns, but often at the cost of being "overconfident" on OOD data.
* **Detection Winner**: The Prototype head provides the most reliable OOD detection at higher shot counts (AUROC 0.78), effectively separating the known from the unknown by focusing on class-central representations.

#### **Slide 6: The "Gaussian Paradox" in High Dimensions**

* **The Findings**: The Gaussian head, theoretically the most sophisticated, performed poorly (AUROC ~0.57).
* **The Theory**: In high-dimensional spaces (512-D CLIP features), Mahalanobis distance is highly sensitive to the "tied covariance" assumption. If the data does not perfectly follow a Gaussian distribution, the model treats normal variations as OOD signals, leading to degraded performance.

#### **Slide 7: Practical Insights: Rejection Trade-offs**

* **Rejection Rates**: Higher shot counts () allow for "tighter" confidence distributions for ID data.
* **Trade-off**: By utilizing a 95% TPR threshold, we can significantly reduce the False Positive Rate (FPR), ensuring that OOD samples are filtered while retaining high classification performance on known data.

#### **Slide 8: Strategic Recommendations**

* **Best Overall**: For systems requiring both classification and safety, the **Prototype Head** at  offers the best balance of calibration, OOD detection, and simplicity.
* **Best for Raw Performance**: The **Linear Probe** is optimal for ID accuracy but requires additional OOD-specific regularization to be used in safety-critical environments.