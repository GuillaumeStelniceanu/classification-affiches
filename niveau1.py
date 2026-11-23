"""Niveau 1 : fusionne les six familles et les metadonnees, puis ecrit submission.csv.

    31 colonnes                          HistGradientBoosting        5 genres
    ------------                         --------------------
    4 x 5 probabilites de classe    ->   200 arbres, prof. 4    ->   argmax
    2 specialistes Blockbuster           poids freq^-0.5             x coefficient
    9 metadonnees de fichier             l2 = 10

TROIS DECISIONS, TOUTES MESUREES.

1. LES METADONNEES. Les CNN redimensionnent toute image en 192x128 des la premiere
   ligne et detruisent sa resolution native. Or celle-ci trahit la source de collecte :
   sur 1 340 affiches en 500x750, UNE SEULE est Art&Essai, contre 63 % dans le format
   810x1080. Ces neuf colonnes valent 5,6 points a elles seules.

2. LA PONDERATION DES CLASSES. Le challenge est note en F1 MACRO : chaque classe pese
   1/5, donc Blockbuster (4,5 % des images) compte autant qu'Art&Essai (30 %). Ponderer
   les classes a l'entrainement du classifieur par freq^-0.5 a rapporte 2,2 points sur
   le classement. La ponderation pleine freq^-1 sur-corrige et fait moins bien.

3. LE COEFFICIENT BLOCKBUSTER. Un F1 est maximal quand precision et rappel s'equilibrent.
   Sans correction, Blockbuster affichait 0,53 de precision pour 0,30 de rappel : le
   modele le sous-predisait. Un coefficient unique sur sa probabilite retablit
   l'equilibre (0,44 / 0,48) et porte son F1 de 0,382 a 0,455.

    python niveau1.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold

from data_utils import CLASSES, N_FOLDS, deduplicate, load_index, test_frame
from features import META_COLS, metadonnees

OUT = Path("outputs")
SEED = 42
ARBRES = 200
# Poids de classe = 1/sqrt(freq). L'exposant 0.5 sort d'un balayage : la ponderation
# pleine 1/freq sur-corrige et fait moins bien, l'absence de ponderation aussi.
# Ecrit ainsi et non `freq**-0.5` : les deux different d'un ulp, et cet ecart suffit
# a faire basculer 5 predictions apres 200 tours de boosting.
FAMILLES = ["resnet192", "resnet288", "sep192", "sep288"]
SPECIALISTES = ["spec192", "spec288"]
GRILLE = np.concatenate([np.linspace(.5, 3., 26), np.linspace(3.2, 6., 15)])


def gbm():
    return HistGradientBoostingClassifier(max_iter=ARBRES, learning_rate=.06,
                                          l2_regularization=10., max_depth=4,
                                          min_samples_leaf=40, random_state=0)


def colonnes(famille, binaire=False):
    return [f"{famille}"] if binaire else [f"{famille}_{c}" for c in CLASSES]


def assemble():
    """Construit la table hors-fold et la table de test, dans le meme ordre de colonnes.

    Hors-fold : chaque ligne vient du seul modele qui n'a pas vu cette image.
    Test : moyenne des 5 modeles de folds — les colonnes y sont donc 2,2 fois moins
    bruitees, ce qui explique qu'une table riche transfere mieux que la validation
    ne le laisse croire.
    """
    df = deduplicate(load_index()).reset_index(drop=True)
    y = df["label"].values
    test = test_frame().reset_index(drop=True)
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    decoupe = list(skf.split(np.zeros(len(y)), y))

    blocs_oof, blocs_test, noms_cols = [], [], []
    for f in FAMILLES + SPECIALISTES:
        binaire = f in SPECIALISTES
        largeur = 2 if binaire else 5
        oof = np.zeros((len(y), largeur))
        for k, (_, i_va) in enumerate(decoupe, 1):
            oof[i_va] = np.load(OUT / f / f"oof{k}.npy")
        tst = np.mean([np.load(OUT / f / f"test{k}.npy") for k in range(1, N_FOLDS + 1)], 0)
        if binaire:                      # on ne garde que P(Blockbuster)
            oof, tst = oof[:, 1:2], tst[:, 1:2]
        blocs_oof.append(oof)
        blocs_test.append(tst)
        noms_cols += colonnes(f, binaire)

    X = pd.DataFrame(np.hstack(blocs_oof), columns=noms_cols)
    Xt = pd.DataFrame(np.hstack(blocs_test), columns=noms_cols)
    X[META_COLS] = metadonnees(df["path"]).values
    Xt[META_COLS] = metadonnees(test["path"]).values
    return X, y, Xt, test["name"].tolist()


def coefficient_blockbuster(P, y):
    """Cherche le multiplicateur qui maximise le F1 macro. Un seul parametre libre."""
    def note(b):
        w = np.ones(len(CLASSES))
        w[CLASSES.index("Blockbuster")] = b
        return f1_score(y, (P * w).argmax(1), average="macro")
    return max(GRILLE, key=note)


def main():
    X, y, Xt, noms = assemble()
    freq = np.bincount(y, minlength=len(CLASSES)) / len(y)
    poids = 1 / np.sqrt(freq)
    poids = poids / poids.sum() * len(CLASSES)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    print(f"{len(X.columns)} colonnes · {len(y)} images · poids de classe "
          f"{dict(zip(CLASSES, np.round(poids, 2)))}")

    # probabilites hors-fold du niveau 1, pour regler le coefficient sans fuite
    P = np.zeros((len(y), len(CLASSES)))
    for i_tr, i_va in cv.split(X, y):
        m = gbm().fit(X.iloc[i_tr].values, y[i_tr], sample_weight=poids[y[i_tr]])
        P[i_va] = m.predict_proba(X.iloc[i_va].values)

    b = coefficient_blockbuster(P, y)
    w = np.ones(len(CLASSES))
    w[CLASSES.index("Blockbuster")] = b
    print(f"coefficient Blockbuster : {b:.2f}")
    print(f"F1 macro hors-fold      : {f1_score(y, (P * w).argmax(1), average='macro'):.4f}"
          f"   (sans coefficient : {f1_score(y, P.argmax(1), average='macro'):.4f})\n")
    print(classification_report(y, (P * w).argmax(1), target_names=CLASSES,
                                digits=3, zero_division=0))

    modele = gbm().fit(X.values, y, sample_weight=poids[y])
    pred = (modele.predict_proba(Xt.values) * w).argmax(1)
    pd.DataFrame({"Id": noms, "Class": pred}).to_csv("submission.csv", index=False)
    print(f"{len(pred)} predictions -> submission.csv")
    for c, n in pd.Series(pred).value_counts().sort_index().items():
        print(f"  {c} {CLASSES[c]:<12} {n:>5}  ({n / len(pred) * 100:4.1f} %)")


if __name__ == "__main__":
    main()
