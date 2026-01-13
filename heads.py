import torch
import torch.nn as nn
import torch.nn.functional as F

class ZeroShotHead:
    def __init__(self, text_features):
        # Ensure text features are normalized
        self.text_features = F.normalize(text_features, dim=1)

    def predict(self, image_features):
        # Enforce normalization on input features
        image_features = F.normalize(image_features, dim=1)
        scores = torch.matmul(image_features, self.text_features.T)
        return scores

class PrototypeHead:
    def fit(self, features, labels):
        # features should be normalized for cosine similarity-based prototypes
        features = F.normalize(features, dim=1)
        self.classes = torch.unique(labels)
        self.prototypes = torch.stack([features[labels==c].mean(0) for c in self.classes])
        self.prototypes = F.normalize(self.prototypes, dim=1)

    def predict(self, features):
        features = F.normalize(features, dim=1)
        return torch.matmul(features, self.prototypes.T)

class LinearProbe:
    def __init__(self, feature_dim, num_classes, lr=1e-3, epochs=100, weight_decay=1e-2):
        self.model = nn.Linear(feature_dim, num_classes)
        self.lr = lr
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        # Added weight decay for better few-shot regularization
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)

    def fit(self, features, labels):
        self.model.to(features.device)
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
    def fit(self, features, labels, shrinkage_ratio=0.5):
        self.classes = torch.unique(labels)
        self.means = torch.stack([features[labels==c].mean(0) for c in self.classes])
        
        # Centering data for tied covariance
        Xc = [features[labels==c] - self.means[i] for i, c in enumerate(self.classes)]
        cov = torch.stack([x.T @ x / len(x) for x in Xc]).mean(0)
        
        # Improved shrinkage logic
        eye = torch.eye(cov.shape[0], device=cov.device)
        cov = (1 - shrinkage_ratio) * cov + shrinkage_ratio * torch.diag(torch.diag(cov))
        
        self.cov_inv = torch.linalg.pinv(cov)

    def predict(self, features):
        scores = []
        for mu in self.means:
            diff = features - mu
            # Negative Mahalanobis distance as a confidence score
            m_dist = (diff @ self.cov_inv * diff).sum(-1)
            scores.append(-m_dist)
        return torch.stack(scores, dim=1)