from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torchvision.transforms as T
import matplotlib.pyplot as plt

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

# ImageNet-O dataset (load locally like ImageNet-Val)
class ImageNetODataset(Dataset):
    def __init__(self, transform=None):
        from datasets import load_dataset
        self.dataset = load_dataset(
            "cais/imagenet-o",
            split="test",
            cache_dir="./hf_cache"
        )
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]["image"]
        # Ensure RGB format (some images might be RGBA or grayscale)
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, -1



# Example usage
if __name__ == "__main__":
    print("Loading...")
    imagenet_val = ImageNetValDataset("data/imagenet-val", transform=transform)
    imagenet_o = ImageNetODataset(transform=transform)
    print(f"ImageNet-Val samples: {len(imagenet_val)}")
    print(f"ImageNet-O samples: {len(imagenet_o)}")
    val_loader = DataLoader(imagenet_val, batch_size=64, shuffle=False, num_workers=4)
    ood_loader = DataLoader(imagenet_o, batch_size=64, shuffle=False, num_workers=4)
    print("Datasets loaded.")
    # visual check
    imgs_val, _ = next(iter(val_loader))
    imgs_ood, _ = next(iter(ood_loader))
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    for i in range(2):
        axes[0, i].imshow(imgs_val[i].permute(1, 2, 0))
        axes[0, i].set_title(f"Val {i}")
        axes[0, i].axis("off")
        axes[1, i].imshow(imgs_ood[i].permute(1, 2, 0))
        axes[1, i].set_title(f"ImageNet-O {i}")
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.show()

