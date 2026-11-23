"""Niveau 0 : entraine les 6 familles de reseaux en validation croisee 5 folds.

Six familles composent le modele final, toutes ecrites a la main et initialisees
aleatoirement :

    famille      architecture              resolution   role
    ---------------------------------------------------------------------------
    resnet192    ResNet-18 (blocs residuels)  192x128    5 classes
    resnet288    ResNet-18                    288x192    5 classes
    sep192       separable + SE               192x128    5 classes
    sep288       separable + SE               288x192    5 classes
    spec192      separable + SE               192x128    binaire Blockbuster
    spec288      separable + SE               288x192    binaire Blockbuster

POURQUOI PLUSIEURS FAMILLES. Mesure : le reseau 288x192 utilise SEUL fait perdre
1,5 point sur le classement, et en fait gagner 1,8 lorsqu'il s'ajoute au 192x128.
Ce n'est pas la qualite individuelle qui compte mais la decorrelation des erreurs :
deux familles quelconques se contredisent sur un quart des images.

CHAQUE FOLD produit les probabilites HORS-FOLD de ses images de validation. Le
niveau 1 ne verra donc jamais la prediction d'un reseau sur une image qu'il a servi
a entrainer — sans cette regle, il recopierait des probabilites a 0,95 d'exactitude
et ignorerait les metadonnees.

Ce script entraine desormais les 6 familles a la suite, dans une seule execution :

    python niveau0.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import data_utils
from data_utils import (CLASSES, N_FOLDS, deduplicate, load_index, oversample_minorities,
                        test_frame, use_cache)
from entrainement import CFG, charge, npy_cache, probas, repris_ou_entraine
from model import ResNet18
from nets import SpecialisteCNN

OUT = Path("outputs")
BLOCKBUSTER = 1

# fabrique, nombre de classes, epochs, mixup
FAMILLES = {
    "resnet192": (lambda: ResNet18(5, width=CFG["width"], dropout=CFG["dropout"]), 5, 30, True),
    "resnet288": (lambda: ResNet18(5, width=CFG["width"], dropout=CFG["dropout"]), 5, 30, True),
    "sep192":    (lambda: SpecialisteCNN(5, width=48, dropout=CFG["dropout"]), 5, 30, True),
    "sep288":    (lambda: SpecialisteCNN(5, width=48, dropout=CFG["dropout"]), 5, 30, True),
    "spec192":   (lambda: SpecialisteCNN(2, width=24, dropout=CFG["dropout"]), 2, 30, False),
    "spec288":   (lambda: SpecialisteCNN(2, width=24, dropout=CFG["dropout"]), 2, 30, False),
}


def prepare(tr, n_classes, cfg):
    """Blockbuster contre le reste pour un specialiste, sinon reequilibrage des classes."""
    if n_classes == 2:
        b = tr.assign(label=(tr["label"] == BLOCKBUSTER).astype(int))
        return pd.concat([b] + [b[b["label"] == 1]] * 4, ignore_index=True)
    return oversample_minorities(tr, seed=cfg["seed"], verbose=False)


def entraine_famille(nom_famille, epochs_override=None, workers=CFG["workers"]):
    """Entraine une famille sur ses 5 folds. Les modeles/probas deja presents sur
    disque sont recharges sans reentrainer (repris_ou_entraine) : relancer ce
    script apres une interruption ne perd rien de ce qui est deja fait."""
    fabrique, n_classes, epochs, avec_mixup = FAMILLES[nom_famille]
    cfg = {**CFG, "workers": workers}
    epochs = epochs_override or epochs
    dossier = OUT / nom_famille

    # equivalent de IMG_H/IMG_W : on fixe la resolution attendue pour cette famille
    attendu = (288, 192) if nom_famille.endswith("288") else (192, 128)
    data_utils.IMG_SIZE = attendu

    print(f"famille {nom_famille} · entree {data_utils.IMG_SIZE} · {epochs} epochs")
    print(f"parametres : {sum(q.numel() for q in fabrique().parameters()) / 1e6:.3f} M")

    df = use_cache(deduplicate(load_index()).reset_index(drop=True))
    y = df["label"].values
    dl_test = charge(use_cache(test_frame().reset_index(drop=True)), False, cfg["batch"])

    oof = np.zeros((len(df), n_classes))
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=cfg["seed"])
    for k, (i_tr, i_va) in enumerate(skf.split(df, y), 1):
        print(f"\n=== FOLD {k}/{N_FOLDS} ===", flush=True)
        dl_va = charge(df.iloc[i_va], False, cfg["batch"])
        m = repris_ou_entraine(dossier / f"modele_fold{k}.pt", fabrique,
                               prepare(df.iloc[i_tr], n_classes, cfg),
                               cfg, epochs, avec_mixup, nom_famille)
        oof[i_va] = npy_cache(dossier / f"oof{k}.npy", lambda: probas(m, dl_va))
        npy_cache(dossier / f"test{k}.npy", lambda: probas(m, dl_test))

    if n_classes == 5:
        print(f"\n{nom_famille} seul : accuracy {(oof.argmax(1) == y).mean():.4f} | "
              f"F1 macro {f1_score(y, oof.argmax(1), average='macro'):.4f}")
    else:
        cible = (y == BLOCKBUSTER).astype(int)
        print(f"\n{nom_famille} : F1 Blockbuster {f1_score(cible, oof.argmax(1)):.4f}")
    print(f"probabilites ecrites dans {dossier}/")


def main():
    for nom_famille in FAMILLES:
        print(f"\n{'=' * 70}\nFAMILLE : {nom_famille}\n{'=' * 70}", flush=True)
        entraine_famille(nom_famille)
    print("\nToutes les familles sont terminees. Tu peux lancer : python niveau1.py")


if __name__ == "__main__":
    main()
