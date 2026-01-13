import torch
import argparse
import open_clip
from torch.utils.data import DataLoader
from dataset_utils import ImageNetValDataset, ImageNetODataset, transform

def cache_features(model_name, dataloader, device, save_path):
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained="openai")
    model = model.to(device)
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            features = model.encode_image(imgs)
            features = torch.nn.functional.normalize(features, dim=1)
            all_features.append(features.cpu())
            all_labels.append(labels)

    torch.save({"features": torch.cat(all_features), "labels": torch.cat(all_labels)}, save_path)
    print(f"Saved {len(all_features)} batches to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ViT-B-16")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--type", choices=["id", "ood"], required=True)
    parser.add_argument("--data_path", type=str, default="data/imagenet-val")
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.type == "id":
        ds = ImageNetValDataset(args.data_path, transform=transform)
    else:
        ds = ImageNetODataset(transform=transform)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    cache_features(args.model, loader, device, args.save_path)