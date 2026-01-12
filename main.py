import torch
from heads import ZeroShotHead, PrototypeHead, LinearProbe, GaussianHead
from utils import sample_k_shots, accuracy, ood_metrics
from text_embed import load_classnames, encode_text


Ks = [0, 1, 2, 4, 8, 16]
seeds = [0, 1, 2]

# Load cached features
id_data = torch.load("cached_features/val_features.pt")
ood_data = torch.load("cached_features/ood_features.pt")

X_all, y_all = id_data["features"], id_data["labels"]
X_ood = ood_data["features"]

# Split data: 80% for k-shot sampling, 20% for evaluation
torch.manual_seed(42)
indices = torch.randperm(len(X_all))
split_idx = int(0.8 * len(X_all))
train_indices = indices[:split_idx]
test_indices = indices[split_idx:]

X_train, y_train = X_all[train_indices], y_all[train_indices]
X_test, y_test = X_all[test_indices], y_all[test_indices]

print(f"Train: {X_train.shape}, Test: {X_test.shape}, OOD: {X_ood.shape}")

results = []

# Zero-shot
classnames = load_classnames("imagenet_classes.txt")
text_features = encode_text(classnames)

zs = ZeroShotHead(text_features)

zs_logits = zs.predict(X_test)
zs_acc = accuracy(zs_logits, y_test)

zs_auroc, zs_fpr = ood_metrics(
    zs_logits.max(1).values.numpy(),
    zs.predict(X_ood).max(1).values.numpy()
)

results.append({
    "K": 0,
    "seed": -1,
    "proto_acc": None,
    "proto_auroc": None,
    "proto_fpr95": None,
    "gauss_acc": None,
    "gauss_auroc": None,
    "gauss_fpr95": None,
    "lin_acc": None,
    "lin_auroc": None,
    "lin_fpr95": None,
    "zs_acc": zs_acc,
    "zs_auroc": zs_auroc,
    "zs_fpr": zs_fpr
})


for K in Ks:
    for seed in seeds:
        print(f"K={K}, seed={seed}")

        # ---- Fit heads ----
        if K > 0:
            Xk, yk = sample_k_shots(X_train, y_train, K, seed)

            proto = PrototypeHead()
            proto.fit(Xk, yk)

            gauss = GaussianHead()
            gauss.fit(Xk, yk)

            lin = LinearProbe(X_train.shape[1], len(torch.unique(y_train)))
            lin.fit(Xk, yk)

        # ---- Evaluate ----
        with torch.no_grad():
            if K == 0:
                continue  # zero-shot handled separately

            proto_logits = proto.predict(X_test)
            gauss_logits = gauss.predict(X_test)
            lin_logits = lin.predict(X_test)

            proto_acc = accuracy(proto_logits, y_test)
            gauss_acc = accuracy(gauss_logits, y_test)
            lin_acc = accuracy(lin_logits, y_test)

            # OOD confidence = max score
            proto_auroc, proto_fpr = ood_metrics(
                proto_logits.max(1).values.cpu().numpy(),
                proto.predict(X_ood).max(1).values.cpu().numpy()
            )

            gauss_auroc, gauss_fpr = ood_metrics(
                gauss_logits.max(1).values.cpu().numpy(),
                gauss.predict(X_ood).max(1).values.cpu().numpy()
            )

            lin_auroc, lin_fpr = ood_metrics(
                lin_logits.max(1).values.cpu().numpy(),
                lin.predict(X_ood).max(1).values.cpu().numpy()
            )

        results.append({
            "K": K,
            "seed": seed,
            "proto_acc": proto_acc,
            "proto_auroc": proto_auroc,
            "proto_fpr95": proto_fpr,
            "gauss_acc": gauss_acc,
            "gauss_auroc": gauss_auroc,
            "gauss_fpr95": gauss_fpr,
            "lin_acc": lin_acc,
            "lin_auroc": lin_auroc,
            "lin_fpr95": lin_fpr,
        })

torch.save(results, "results.pt")
