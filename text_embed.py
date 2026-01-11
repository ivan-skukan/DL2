import torch
from transformers import CLIPTokenizer, CLIPTextModel

PROMPTS = [
    "a photo of a {}",
    "a photo of the {}",
]

def load_classnames(path):
    with open(path, "r") as f:
        return [line.strip() for line in f]

def encode_text(
    classnames,
    model_name="openai/clip-vit-base-patch16",
    device="cuda",
    save_path="text_features.pt",
):
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    text_model = CLIPTextModel.from_pretrained(model_name).to(device)
    text_model.eval()

    all_features = []

    with torch.no_grad():
        for name in classnames:
            prompts = [p.format(name) for p in PROMPTS]
            inputs = tokenizer(
                prompts,
                padding=True,
                return_tensors="pt"
            ).to(device)

            outputs = text_model(**inputs)
            features = outputs.pooler_output
            features = torch.nn.functional.normalize(features, dim=1)

            # prompt ensembling (mean)
            mean_feat = features.mean(dim=0)
            mean_feat = mean_feat / mean_feat.norm()

            all_features.append(mean_feat)

    text_features = torch.stack(all_features)
    torch.save(text_features.cpu(), save_path)

    print(f"Saved text features: {text_features.shape}")
    return text_features
