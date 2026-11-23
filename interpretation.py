"""Ouvrir la boite : sur quoi le classifieur final s'appuie-t-il reellement ?

Un empilement a trois familles risque de devenir inexplicable. Ce script mesure,
sans supposition, ce que le niveau 1 utilise vraiment.

Trois mesures complementaires :

1. IMPORTANCE PAR PERMUTATION. On melange aleatoirement une colonne et on regarde
   combien le F1 macro chute. Contrairement aux importances internes d'un arbre, qui
   surestiment les variables a nombreuses valeurs distinctes, cette mesure repond a
   la seule question qui compte : de quoi le modele a-t-il besoin ?

2. ABLATION PAR BLOC. Retirer une famille entiere et remesurer. L'importance par
   permutation sous-estime les colonnes redondantes — si deux reseaux disent la meme
   chose, en brouiller un seul ne coute rien. Retirer le bloc complet le revele.

3. QUI DECIDE QUOI. Pour chaque classe, la part des predictions ou le classifieur
   suit le vote majoritaire des CNN, et celle ou il s'en ecarte. C'est la mesure qui
   dit si le classifieur arbitre vraiment ou s'il se contente de recopier.

    python interpretation.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from data_utils import CLASSES
from features import META_COLS
from niveau1 import OUT, SEED, assemble, gbm

CNN192 = [f"resnet192_{c}" for c in CLASSES]
CNN288 = [f"resnet288_{c}" for c in CLASSES]
SEP = [f"sep192_{c}" for c in CLASSES]
SEP288 = [f"sep288_{c}" for c in CLASSES]
SPEC = ["spec192", "spec288"]
BLOCS = {"ResNet 192x128": CNN192, "ResNet 288x192": CNN288,
         "Separable 192x128": SEP, "Separable 288x192": SEP288,
         "Specialiste Blockbuster": SPEC, "Metadonnees": META_COLS}


def main():
    X, y, Xt, _ = assemble()
    blocs = {f: X[[c for c in X.columns if c.startswith(f + '_')]].values
             for f in ('resnet192', 'resnet288', 'sep192', 'sep288')}
    cols = CNN192 + CNN288 + SEP + SEP288 + SPEC + META_COLS
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    print("=" * 74)
    print("1. IMPORTANCE PAR PERMUTATION  (chute de F1 MACRO quand la colonne est brouillee)")
    print("=" * 74)
    Xtr, Xte, ytr, yte = train_test_split(X[cols].values, y, test_size=.3,
                                          stratify=y, random_state=0)
    m = gbm().fit(Xtr, ytr)
    imp = permutation_importance(m, Xte, yte, n_repeats=8, random_state=0, n_jobs=1,
                                 scoring='f1_macro')
    ordre = np.argsort(imp.importances_mean)[::-1]
    for i in ordre[:14]:
        barre = "#" * max(1, int(imp.importances_mean[i] * 400))
        print(f"  {cols[i]:<22} {imp.importances_mean[i]:+.4f} ± {imp.importances_std[i]:.4f}  {barre}")
    print(f"  ... {len(cols) - 14} colonnes restantes sous {imp.importances_mean[ordre[13]]:.4f}")

    print()
    print("=" * 74)
    print("2. ABLATION PAR BLOC  (retirer une famille entiere)")
    print("=" * 74)
    ref = (cross_val_predict(gbm(), X[cols].values, y, cv=cv) == y).mean()
    print(f"  modele complet ({len(cols)} colonnes)                acc {ref:.4f}")
    for nom, bloc in BLOCS.items():
        restant = [c for c in cols if c not in bloc]
        acc = (cross_val_predict(gbm(), X[restant].values, y, cv=cv) == y).mean()
        print(f"  sans {nom:<28} ({len(restant):>2} col.) acc {acc:.4f}   "
              f"cout {ref - acc:+.4f}")

    print()
    print("=" * 74)
    print("3. LE CLASSIFIEUR ARBITRE-T-IL, OU RECOPIE-T-IL ?")
    print("=" * 74)
    pred = cross_val_predict(gbm(), X[cols].values, y, cv=cv)
    votes = np.stack([v.argmax(1) for v in blocs.values()])
    majorite = np.array([np.bincount(v, minlength=len(CLASSES)).argmax() for v in votes.T])
    accord_total = (votes == votes[0]).all(0)
    suit = pred == majorite

    print(f"  les 4 reseaux sont unanimes           : {accord_total.mean() * 100:5.1f} % des images")
    print(f"    -> le classifieur les suit          : {suit[accord_total].mean() * 100:5.1f} %")
    print(f"    -> accuracy sur ces images          : {(pred == y)[accord_total].mean():.4f}")
    print()
    print(f"  les reseaux sont divises              : {(~accord_total).mean() * 100:5.1f} % des images")
    print(f"    -> le classifieur suit la majorite  : {suit[~accord_total].mean() * 100:5.1f} %")
    print(f"    -> il tranche CONTRE la majorite    : {(1 - suit[~accord_total].mean()) * 100:5.1f} %")
    print(f"    -> accuracy sur ces images          : {(pred == y)[~accord_total].mean():.4f}")
    print()
    contre = ~suit & ~accord_total
    gain = (pred == y)[contre].mean() - (majorite == y)[contre].mean()
    print(f"  Quand il contredit le vote majoritaire ({contre.sum()} images) :")
    print(f"    accuracy du vote majoritaire        : {(majorite == y)[contre].mean():.4f}")
    print(f"    accuracy du classifieur             : {(pred == y)[contre].mean():.4f}")
    print(f"    gain net de l'arbitrage             : {gain:+.4f}")


if __name__ == "__main__":
    main()
