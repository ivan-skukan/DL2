from datasets import load_dataset

print("Loading ImageNet-1k validation set...")
try:
    imagenet_val = load_dataset("data/imagenet-1k", split="validation")
    print(f"Number of samples in ImageNet-1k validation set: {len(imagenet_val)}")

#print("Loading ImageNet-O (only val exists)...")
#try:
 #   dataset = load_dataset("cais/imagenet-o")
 #   print(f"Available splits: {dataset.keys()}")
 #   for split_name in dataset.keys():
  #      print(f"  {split_name}: {len(dataset[split_name])} samples")
#except Exception as e:
 #   print(f"Error loading all splits: {e}")


