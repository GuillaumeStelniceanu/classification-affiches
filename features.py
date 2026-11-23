"""Metadonnees d'image : tout ce que le CNN jette en redimensionnant en 192x128.

Mesure sur le train (5-fold, aucune image decodee) : ces seules colonnes donnent
0.4366 d'accuracy contre 0.3074 pour la classe majoritaire, et surtout F1 0.703 sur
Art&Essai. Les affiches Art&Essai viennent d'une source differente : sur les 1340
images en 500x750, une seule est Art&Essai, alors que le 810x1080 en contient 63 %.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

META_COLS = ["largeur", "hauteur", "ratio", "pixels", "octets", "octets_par_pixel",
             "log_pixels", "est_500x750", "est_1080_haut"]


def metadonnees(paths):
    """DataFrame des features calculables sans decoder l'image.

    `paths` doit pointer sur les fichiers d'ORIGINE, pas sur le cache : le cache est
    uniformise en 256x384 et a donc perdu toute l'information de resolution.
    """
    lignes = []
    for p in paths:
        p = Path(p)
        w, h = Image.open(p).size
        octets = p.stat().st_size
        lignes.append((w, h, h / w, w * h, octets, octets / (w * h),
                       np.log10(w * h), int(w == 500 and h == 750), int(h == 1080)))
    return pd.DataFrame(lignes, columns=META_COLS)
