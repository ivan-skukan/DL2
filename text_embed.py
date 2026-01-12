import torch
import open_clip

# Prompts for ensembling
PROMPTS = [
    "a photo of a {}",
    "a photo of the {}",
]

def load_classnames(path):
    """
    Load class names from a file and remove numeric prefixes.
    Expects lines like:
        948, Granny_Smith
        949, strawberry
    Returns:
        list of strings: ['Granny_Smith', 'strawberry', ...]
    """
    classnames = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split by first comma, take second part, strip spaces
            name = line.split(",", 1)[1].strip()
            classnames.append(name)
    return classnames

def encode_text(
    classnames,
    model_name="ViT-B-16",
    device=None,
    save_path="text_features.pt",
):
    """
    Encode a list of class names into CLIP text embeddings using open_clip.
    Uses prompt ensembling and returns normalized features.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load open_clip model (text encoder is tied to the image backbone)
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="openai")
    model = model.to(device)
    model.eval()

    all_features = []

    with torch.no_grad():
        for name in classnames:
            # Create multiple prompts for ensembling
            prompts = [p.format(name) for p in PROMPTS]

            # open_clip.tokenize returns tensor of token ids
            tokenized = open_clip.tokenize(prompts).to(device)
            text_features = model.encode_text(tokenized)           # [num_prompts, hidden_dim]
            text_features = torch.nn.functional.normalize(text_features, dim=1)

            # prompt ensembling: mean vector across prompts
            mean_feat = text_features.mean(dim=0)
            mean_feat = mean_feat / mean_feat.norm()

            all_features.append(mean_feat)

    text_features = torch.stack(all_features)  # [num_classes, hidden_dim]
    torch.save(text_features.cpu(), save_path)
    print(f"Saved text features: {text_features.shape}")
    return text_features

# Example usage
if __name__ == "__main__":
    classnames = load_classnames("imagenet_classes.txt")
    print(f"Loaded {len(classnames)} class names.")
    encode_text(
        classnames,
        model_name="ViT-B-16",
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_path="cached_features/text_features.pt",
    )