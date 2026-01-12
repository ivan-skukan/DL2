import torch
from torch.utils.data import DataLoader
import open_clip
from dataset_utils import ImageNetValDataset, ImageNetODataset, transform

def cache_features(model_name, dataloader, device="cuda", save_path="features.pt"):
    # Load open_clip model
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained="openai")
    model = model.to(device)
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            features = model.encode_image(imgs)
            features = torch.nn.functional.normalize(features, dim=1)  # normalize for cosine similarity
            all_features.append(features.cpu())
            all_labels.append(labels)

    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)
    torch.save({"features": all_features, "labels": all_labels}, save_path)
    print(f"Saved {len(all_features)} features to {save_path}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    batch_size = 64
    model_name = "ViT-B-16"  # open_clip format

    # Datasets + loaders
    # imagenet_val = ImageNetValDataset("data/imagenet-val", transform=transform)
    imagenet_o = ImageNetODataset(transform=transform)
    # val_loader = DataLoader(imagenet_val, batch_size=batch_size, shuffle=False, num_workers=4)
    ood_loader = DataLoader(imagenet_o, batch_size=batch_size, shuffle=False, num_workers=4)

    # Cache features
    # cache_features(model_name, val_loader, device=device, save_path="cached_features/val_features.pt")
    cache_features(model_name, ood_loader, device=device, save_path="cached_features/ood_features.pt")
