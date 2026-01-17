from pydoc import classname
from tkinter import Image
import torch
import open_clip

PROMPTS = [
    "a photo of a {}",
    "a photo of the {}",
]

def load_classnames_raw(path):
    """
    Loads class names from file like:
        0, tench
        1, goldfish
    Returns:
        ['tench', 'goldfish', ...]
    Order here does NOT matter.
    """
    names = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name = line.split(",", 1)[1].strip()
            names.append(name)
    return names


def reorder_classnames(classnames_raw, class_to_idx):
    """
    Reorder class names to EXACTLY match dataset labels.
    This guarantees compatibility with cached image embeddings.
    """
    ordered = [None] * len(class_to_idx)
    for name in classnames_raw:
        idx = class_to_idx[name]
        ordered[idx] = name

    assert all(x is not None for x in ordered)
    return ordered


def encode_text(
    classnames,
    model_name="ViT-B-16",
    device=None,
    save_path="cached_features/text_features.pt",
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained="openai"
    )
    model = model.to(device)
    model.eval()

    all_features = []

    with torch.no_grad():
        for name in classnames:
            prompts = [p.format(name) for p in PROMPTS]
            tokens = open_clip.tokenize(prompts).to(device)

            feats = model.encode_text(tokens)
            feats = torch.nn.functional.normalize(feats, dim=1)

            feat = feats.mean(dim=0)
            feat = feat / feat.norm()
            all_features.append(feat)

    text_features = torch.stack(all_features)
    torch.save(text_features.cpu(), save_path)

    print(f"Saved text features: {text_features.shape}")
    return text_features
if __name__ == "__main__":
    from dataset_utils import ImageNetValDataset
    dataset = ImageNetValDataset("data/imagenet-val")
    classnames = load_classnames_raw("imagenet_classes.txt")
    
    # The classnames file is already ordered 0-999, matching alphabetically sorted WordNet IDs
    print(f"Loaded {len(classnames)} class names")
    print(f"Dataset has {len(dataset.class_to_idx)} classes")
    
    encode_text(
        classnames,
        model_name="ViT-B-16",
        device="cpu",
        save_path="cached_features/text_features.pt",
    )