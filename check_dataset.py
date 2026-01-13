import torch
from torch.utils.data import DataLoader
from dataset_utils import ImageNetValDataset, ImageNetODataset, transform

def main():
    print("--- Dataset Diagnostic Check ---")
    
    # Check ID Dataset
    try:
        id_ds = ImageNetValDataset("data/imagenet-val", transform=transform)
        print(f"ID (ImageNet-Val) found: {len(id_ds)} samples.")
        id_loader = DataLoader(id_ds, batch_size=4)
        imgs, labels = next(iter(id_loader))
        print(f"ID Batch Shape: {imgs.shape}, Labels: {labels}")
    except Exception as e:
        print(f"ID Dataset Error: {e}. Check if 'data/imagenet-val' exists.")

    # Check OOD Dataset
    try:
        ood_ds = ImageNetODataset(transform=transform)
        print(f"OOD (ImageNet-O) found: {len(ood_ds)} samples.")
        ood_loader = DataLoader(ood_ds, batch_size=4)
        imgs, _ = next(iter(ood_loader))
        print(f"OOD Batch Shape: {imgs.shape}")
    except Exception as e:
        print(f"OOD Dataset Error: {e}")

if __name__ == "__main__":
    main()