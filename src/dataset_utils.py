from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset
import torchvision.transforms as T

# Standard CLIP normalization
transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711])
])

class ImageNetValDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"Directory {root} not found.")
            
        # Standard folder-per-class structure
        # Create 0-indexed mapping from sorted directory names
        self.class_to_idx = {cls.name: i for i, cls in enumerate(sorted(d for d in root.iterdir() if d.is_dir()))}
        for cls_dir, idx in self.class_to_idx.items():
            for img_path in (root / cls_dir).iterdir():
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.samples.append((img_path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

class ImageNetODataset(Dataset):
    def __init__(self, transform=None):
        self.dataset = load_dataset("cais/imagenet-o", split="test", cache_dir="./hf_cache")
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]["image"].convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, -1