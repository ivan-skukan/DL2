import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel

def cache_features(model_name, dataloader, device="cuda", save_path="features.pt"):
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            inputs = processor(images=imgs, return_tensors="pt").to(device)
            features = model.get_image_features(**inputs)
            all_features.append(features.cpu())
            all_labels.append(labels)

    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)
    torch.save({"features": all_features, "labels": all_labels}, save_path)
    print(f"Saved {len(all_features)} features to {save_path}")
