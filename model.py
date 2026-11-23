"""ResNet-18 écrit à la main : aucune architecture importée, aucun poids pré-entraîné."""
import torch.nn as nn


class Block(nn.Module):
    """Bloc résiduel à 2 convolutions 3x3 (BasicBlock)."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))
        self.skip = nn.Identity() if stride == 1 and cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        self.act = nn.ReLU(True)

    def forward(self, x):
        return self.act(self.f(x) + self.skip(x))


class ResNet18(nn.Module):
    def __init__(self, n_classes=5, width=64, dropout=.3):
        super().__init__()
        widths = [width, width * 2, width * 4, width * 8]
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 7, 2, 3, bias=False),
                                  nn.BatchNorm2d(widths[0]), nn.ReLU(True), nn.MaxPool2d(3, 2, 1))
        blocks, cin = [], widths[0]
        for i, c in enumerate(widths):
            blocks += [Block(cin, c, stride=1 if i == 0 else 2), Block(c, c)]
            cin = c
        self.body = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(cin, n_classes))
        for m in self.modules():
            if isinstance(m, Block):
                nn.init.zeros_(m.f[-1].weight)

    def forward(self, x):
        return self.head(self.body(self.stem(x)))
