# CLIP Analysis

Analysis of CLIP metrics with different classification heads.

Datasets: 
- ImageNet-1k for ID
- ImageNet-O for OOD

Backbone: ViT-B/16

## Setup

### 1. Data Preparation
- Download the **ImageNet-1k** validation set.
- **ImageNet-O** will be handled automatically via the Hugging Face library.
- Place them in a folder for datasets. You need approximately 7GB for both.

### 2. Environment Setup with uv
This project uses `uv` for fast, reliable package management.

**Install uv** (if not already installed):
```bash
# macOS/Linux
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

# Windows (PowerShell)
powershell -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"
Initialize and Activate Environment:

Bash
# Create the virtual environment
uv venv

# Activate the environment
source .venv/bin/activate
Install Dependencies: Since the project is initialized with pyproject.toml, you can simply run:

Bash
uv sync
3. Embeddings
You will need embeddings for the datasets. You can create them using embed.py or download them from: https://drive.google.com/drive/folders/1qCR5HqHE9rxMuUXGp1df7QTSLGKt2L4u?usp=sharing

Store them in a directory named cached_features/.

Running the Project
Sanity Check: Run python check_dataset.py to ensure datasets are accessible.
Feature Extraction: If not downloaded, run python embed.py for ID and OOD features.
Text Embedding: Run python text_embed.py to generate class-name features.
Main Experiment: Run python main.py to evaluate heads and generate results.pt.
Visualization: Run python plot_results.py to generate performance plots.

### Next Step: Step 1 - Sanity Checks
Now that your environment is ready, try running the diagnostic script again:
```bash
python check_dataset.py
This will verify if the local ImageNet-1k files and the ImageNet-O (via HF) are loading correctly. If you encounter a FileNotFoundError for ImageNet-1k, ensure your path in check_dataset.py matches where you stored the data.