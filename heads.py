import torch
import torch.nn as nn

class ZeroShotHead:
    def __init__(self, text_features):
        self.text_features = text_features  # [num_classes, dim]

    def predict(self, image_features):
        # cosine similarity
        scores = torch.matmul(image_features, self.text_features.T)
        return scores

class PrototypeHead:
    def fit(self, features, labels):
        self.classes = torch.unique(labels)
        self.prototypes = torch.stack([features[labels==c].mean(0) for c in self.classes])

    def predict(self, features):
        return torch.matmul(features, self.prototypes.T)

class LinearProbe:
    def __init__(self, feature_dim, num_classes, lr=1e-3, epochs=100):
        self.model = nn.Linear(feature_dim, num_classes)
        self.lr = lr
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, features, labels):
        self.model.train()
        for _ in range(self.epochs):
            logits = self.model(features)
            loss = self.criterion(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, features):
        self.model.eval()
        with torch.no_grad():
            return self.model(features)

class GaussianHead:
    def fit(self, features, labels, shrinkage=True):
        self.classes = torch.unique(labels)
        self.means = torch.stack([features[labels==c].mean(0) for c in self.classes])
        Xc = [features[labels==c] - self.means[i] for i,c in enumerate(self.classes)]
        cov = torch.stack([x.T @ x / len(x) for x in Xc]).mean(0)
        if shrinkage:
            # simple Ledoit-Wolf style shrinkage to diagonal
            cov = 0.5 * cov + 0.5 * torch.diag(torch.diag(cov))
        self.cov_inv = torch.linalg.pinv(cov)

    def predict(self, features):
        # Mahalanobis distance as negative score
        scores = []
        for mu in self.means:
            diff = features - mu
            m_dist = (diff @ self.cov_inv * diff).sum(-1)
            scores.append(-m_dist)
        return torch.stack(scores, dim=1)
