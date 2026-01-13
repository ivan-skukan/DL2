# CLIP Analysis

Analysis of CLIP metrics with different classification heads.

Datasets: 
- ImageNet-1k for ID
- ImageNet-O for OOD

Backbone: ViT-B/16


## Setup

Download ImageNet-1k validation set. ImageNet-O will be used through HF library. Place them in a folder for datasets. You need about 7GB for both.

Create and activate the virtual environment:
```bash
# <UNIX>
python3 -m venv <venv name>
source ./<venv name>/bin/activate
# <\UNIX>

# <WINDOWS>
python -m venv <venv name>
./<venv name>/Scripts/activate.ps1
# <\WINDOWS>
```

Install required Python libraries:

```bash
pip install -r requirements.txt
```

You will need embeddings for the datasets. Those can be created with the embed.py file or downloaded from

```bash
https://drive.google.com/drive/folders/1qCR5HqHE9rxMuUXGp1df7QTSLGKt2L4u?usp=sharing
```
Store them in cached_features
