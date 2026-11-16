"""Models to attack: shallow MLP, CNN, small Transformer encoder (ViT).
Every model starts with a Normalize layer so it takes raw [0, 1] pixels."""
import torch
import torch.nn as nn


class Normalize(nn.Module):
    def __init__(self, mean=0.1307, std=0.3081):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class MLP(nn.Module):
    def __init__(self, hidden=512, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, 10))

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
                nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
                nn.MaxPool2d(2))

        self.features = nn.Sequential(block(1, 32), block(32, 64))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                        nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(128, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


class ViT(nn.Module):
    """7x7 patches -> 16 tokens + [CLS] -> 4 Transformer encoder layers."""
    def __init__(self, patch=7, dim=64, depth=4, heads=4, mlp_ratio=2, dropout=0.1):
        super().__init__()
        n_patches = (28 // patch) ** 2
        self.patch_embed = nn.Conv2d(1, dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * mlp_ratio, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1) + self.pos_embed
        return self.head(self.norm(self.encoder(x)[:, 0]))


ARCHS = {"mlp": MLP, "cnn": CNN, "vit": ViT}


def build_model(arch: str) -> nn.Module:
    return nn.Sequential(Normalize(), ARCHS[arch]())


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
