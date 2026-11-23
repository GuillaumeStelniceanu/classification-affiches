"""Specialiste Blockbuster : CNN compact ecrit a la main, aucun poids pre-entraine.

Pourquoi une architecture differente du ResNet-18 du modele principal :

- 151 images positives seulement. Un ResNet-18, meme en width 32 (2.8 M parametres),
  memorise ce volume instantanement. Ce reseau tient en ~0.2 M parametres.
- Les convolutions separables en profondeur (depthwise) coutent ~8x moins de parametres
  qu'une convolution 3x3 pleine a nombre de canaux egal.
- Les blocs Squeeze-Excitation ponderent les canaux a partir d'une moyenne globale.
  C'est exactement la forme d'indice qui separe ces classes : la mesure sur le train
  montre que la luminosite (0.302 Horreur -> 0.500 Comedie) et la saturation portent
  l'essentiel du signal, or ce sont des statistiques globales, pas des motifs locaux.
"""
import torch.nn as nn


class SE(nn.Module):
    """Squeeze-Excitation : repondere chaque canal selon sa moyenne spatiale."""

    def __init__(self, c, reduction=8):
        super().__init__()
        cache = max(4, c // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(c, cache), nn.ReLU(True),
                                nn.Linear(cache, c), nn.Sigmoid())

    def forward(self, x):
        p = self.fc(self.pool(x).flatten(1))
        return x * p[:, :, None, None]


class BlocSeparable(nn.Module):
    """Convolution 3x3 depthwise + projection 1x1 + excitation de canaux."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
            nn.BatchNorm2d(cin), nn.ReLU(True),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(True),
            SE(cout))

    def forward(self, x):
        return self.f(x)


class SpecialisteCNN(nn.Module):
    """~0.2 M parametres. Sortie binaire par defaut (Blockbuster contre le reste)."""

    def __init__(self, n_classes=2, width=16, dropout=.4):
        super().__init__()
        w = [width, width * 2, width * 4, width * 8]
        self.stem = nn.Sequential(nn.Conv2d(3, w[0], 5, 2, 2, bias=False),
                                  nn.BatchNorm2d(w[0]), nn.ReLU(True))
        corps, cin = [], w[0]
        for i, c in enumerate(w):
            corps += [BlocSeparable(cin, c, stride=1 if i == 0 else 2),
                      BlocSeparable(c, c)]
            cin = c
        self.corps = nn.Sequential(*corps)
        self.tete = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(cin, n_classes))

    def forward(self, x):
        return self.tete(self.corps(self.stem(x)))
