from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torchvision.transforms as T

# Shared image transforms
transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711])
])

# ImageNet-1k validation dataset
class ImageNetValDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []
        root = Path(root)
        self.class_to_idx = {cls.name: i for i, cls in enumerate(sorted(root.iterdir()))}
        for cls_dir in root.iterdir():
            if cls_dir.is_dir():
                for img_path in cls_dir.iterdir():
                    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        self.samples.append((img_path, self.class_to_idx[cls_dir.name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ImageNet-O dataset via Hugging Face
class ImageNetODataset(Dataset):
    def __init__(self, transform=None):
        self.dataset = load_dataset("Voxel51/ImageNet-O", split="test")
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # OOD labels can just be -1 or 0 if needed
        return img, -1


# Example usage
if __name__ == "__main__":
    print("Loading...")
    imagenet_val = ImageNetValDataset("data/imagenet-val", transform=transform)
    imagenet_o = ImageNetODataset(transform=transform)

    val_loader = DataLoader(imagenet_val, batch_size=64, shuffle=False, num_workers=4)
    ood_loader = DataLoader(imagenet_o, batch_size=64, shuffle=False, num_workers=4)

    print("Datasets loaded.")