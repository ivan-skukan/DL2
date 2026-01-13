import torch
import argparse
import os
from heads import ZeroShotHead, PrototypeHead, LinearProbe, GaussianHead
from utils import sample_k_shots, accuracy, ood_metrics
from calibrate import temperature_scale, choose_ood_threshold

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load cached features
    id_data = torch.load(args.id_features)
    ood_data = torch.load(args.ood_features)
    text_features = torch.load(args.text_features).to(device)

    X_all, y_all = id_data["features"], id_data["labels"]
    X_ood = ood_data["features"].to(device)

    # Consistent split: 80% for k-shot sampling/fit, 20% for testing
    torch.manual_seed(args.split_seed)
    indices = torch.randperm(len(X_all))
    split_idx = int(0.8 * len(X_all))
    X_train, y_train = X_all[indices[:split_idx]].to(device), y_all[indices[:split_idx]].to(device)
    X_test, y_test = X_all[indices[split_idx:]].to(device), y_all[indices[split_idx:]].to(device)

    num_classes = int(y_all.max().item()) + 1
    results = []

    # Zero-shot baseline
    zs = ZeroShotHead(text_features)
    zs_logits = zs.predict(X_test)
    zs_ood_logits = zs.predict(X_ood)
    
    # Calibration example: setting a threshold on zs confidence
    zs_conf = zs_logits.max(1).values
    tau = choose_ood_threshold(zs_conf, torch.ones_like(y_test, dtype=torch.bool), percentile=5)
    
    zs_auroc, zs_fpr = ood_metrics(zs_conf, zs_ood_logits.max(1).values)
    results.append({
        "K": 0, "seed": -1, "head": "ZeroShot",
        "acc": accuracy(zs_logits, y_test), "auroc": zs_auroc, "fpr95": zs_fpr
    })

    # K-Shot experiments
    for K in args.ks:
        for seed in args.seeds:
            print(f"K={K}, seed={seed}")
            Xk, yk = sample_k_shots(X_train, y_train, K, seed)

            heads = {
                "Prototype": PrototypeHead(),
                "Gaussian": GaussianHead(),
                "LinearProbe": LinearProbe(X_train.shape[1], num_classes)
            }

            for name, head in heads.items():
                if name == "Gaussian":
                    head.fit(Xk, yk, shrinkage_ratio=args.shrinkage)
                else:
                    head.fit(Xk, yk)

                test_logits = head.predict(X_test)
                ood_logits = head.predict(X_ood)

                acc = accuracy(test_logits, y_test)
                auroc, fpr = ood_metrics(test_logits.max(1).values, ood_logits.max(1).values)

                results.append({
                    "K": K, "seed": seed, "head": name,
                    "acc": acc, "auroc": auroc, "fpr95": fpr
                })

    torch.save(results, args.output)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id_features", type=str, default="cached_features/val_features.pt")
    parser.add_argument("--ood_features", type=str, default="cached_features/ood_features.pt")
    parser.add_argument("--text_features", type=str, default="cached_features/text_features.pt")
    parser.add_argument("--output", type=str, default="results.pt")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--shrinkage", type=float, default=0.5)
    main(parser.parse_args())