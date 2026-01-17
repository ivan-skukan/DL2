import torch
import argparse
from .heads import ZeroShotHead, PrototypeHead, LinearProbe, GaussianHead
from .utils import sample_k_shots, accuracy, ood_metrics
from .calibrate import tune_temperature, temperature_scale, expected_calibration_error, plot_reliability_diagram
from .visualize import plot_confidence_histograms, plot_pr_curve, plot_feature_embeddings, plot_retained_acc_vs_rejection

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    id_data = torch.load(args.id_features)
    ood_data = torch.load(args.ood_features)
    text_features = torch.load(args.text_features).to(device)
    X_all, y_all = id_data["features"], id_data["labels"]
    X_ood = ood_data["features"].to(device)

    # Split
    torch.manual_seed(args.split_seed)
    indices = torch.randperm(len(X_all))
    split_idx = int(0.8 * len(X_all))
    X_train, y_train = X_all[indices[:split_idx]].to(device), y_all[indices[:split_idx]].to(device)
    X_test, y_test = X_all[indices[split_idx:]].to(device), y_all[indices[split_idx:]].to(device)

    # CRITICAL: Filter both training and test sets for valid labels
    # num_classes is defined as text_features.shape[0]
    num_classes = text_features.shape[0]

    train_mask = (y_train >= 0) & (y_train < num_classes)
    X_train, y_train = X_train[train_mask], y_train[train_mask]

    test_mask = (y_test >= 0) & (y_test < num_classes)
    X_test, y_test = X_test[test_mask], y_test[test_mask]
    results = []

    # Experiment Loop
    for K in [0] + args.ks:
        for seed in (args.seeds if K > 0 else [0]):
            print(f"--- K={K}, seed={seed} ---")
            
            # Setup Head
            if K == 0:
                name, head = "ZeroShot", ZeroShotHead(text_features)
            else:
                Xk, yk = sample_k_shots(X_train, y_train, K, seed)
                # For this example, let's just pick one head to visualize deeply 
                # or loop through Prototype, Gaussian, LinearProbe
                heads_to_run = {
                    "Prototype": PrototypeHead(),
                    "Gaussian": GaussianHead(),
                    "LinearProbe": LinearProbe(X_train.shape[1], num_classes)
                }
                
            for name, head in ({"ZeroShot": head}.items() if K==0 else heads_to_run.items()):
                if name == "Gaussian" and K > 0:
                    head.fit(Xk, yk, shrinkage_ratio=args.shrinkage)
                elif K > 0:
                    head.fit(Xk, yk)

                test_logits = head.predict(X_test)
                ood_logits = head.predict(X_ood)

                # Calibrate
                T, _ = tune_temperature(test_logits, y_test, name=f"{name}_K{K}", device=device)
                ece = expected_calibration_error(test_logits, y_test, T=T)
                
                # Metrics
                test_conf = torch.softmax(temperature_scale(test_logits, T), dim=1).max(1).values
                ood_conf = torch.softmax(temperature_scale(ood_logits, T), dim=1).max(1).values
                auroc, fpr = ood_metrics(test_conf, ood_conf)
                acc = accuracy(test_logits, y_test)

                # Visualize only first seed of each K
                if seed == 0:
                    exp_name = f"{name}_K{K}"
                    plot_reliability_diagram(test_logits, y_test, T=T, name=exp_name)
                    plot_confidence_histograms(test_conf, ood_conf, name=exp_name)
                    plot_pr_curve(test_conf, ood_conf, name=exp_name)
                    plot_retained_acc_vs_rejection(test_conf, y_test, test_logits.argmax(1), name=exp_name)
                    if K == 16: # Only run t-SNE for high K to save time
                        plot_feature_embeddings(X_test, X_ood, y_test, name=exp_name)

                results.append({"K": K, "seed": seed, "head": name, "acc": acc, "auroc": auroc, "fpr95": fpr, "ece": ece})

    torch.save(results, args.output)
    print(f"Final results saved to {args.output}")

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