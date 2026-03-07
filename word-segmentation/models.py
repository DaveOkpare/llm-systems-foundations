import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits
