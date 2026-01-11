import torch
from heads import ZeroShotHead, PrototypeHead, LinearProbe, GaussianHead
from utils import sample_k_shots, accuracy, ood_metrics
from text_embed import load_classnames, encode_text


Ks = [0, 1, 2, 4, 8, 16]
seeds = [0, 1, 2]

# Load cached features
id_data = torch.load("val_features.pt")
ood_data = torch.load("ood_features.pt")

X_id, y_id = id_data["features"], id_data["labels"]
X_ood = ood_data["features"]

# Normalize once (important for cosine-based heads)
X_id = torch.nn.functional.normalize(X_id, dim=1)
X_ood = torch.nn.functional.normalize(X_ood, dim=1)

results = []

# Zero-shot
classnames = load_classnames("imagenet_classes.txt")
text_features = encode_text(classnames)

zs = ZeroShotHead(text_features)

zs_logits = zs.predict(X_id)
zs_acc = accuracy(zs_logits, y_id)

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
            Xk, yk = sample_k_shots(X_id, y_id, K, seed)

            proto = PrototypeHead()
            proto.fit(Xk, yk)

            gauss = GaussianHead()
            gauss.fit(Xk, yk)

            lin = LinearProbe(X_id.shape[1], len(torch.unique(y_id)))
            lin.fit(Xk, yk)

        # ---- Evaluate ----
        with torch.no_grad():
            if K == 0:
                continue  # zero-shot handled separately

            proto_logits = proto.predict(X_id)
            gauss_logits = gauss.predict(X_id)
            lin_logits = lin.predict(X_id)

            proto_acc = accuracy(proto_logits, y_id)
            gauss_acc = accuracy(gauss_logits, y_id)
            lin_acc = accuracy(lin_logits, y_id)

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
