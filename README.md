# Classification d'affiches de films

Projet scolaire : prédire le genre d'une affiche de film parmi 5 classes
(Animation, Blockbuster, Horreur, Comédie, Art&Essai). Le challenge est noté en **F1 macro**.

Contrainte : tous les réseaux sont **écrits à la main et entraînés from scratch**
(aucune architecture importée, aucun poids pré-entraîné).

## Approche

Un empilement (stacking) à deux niveaux :

**Niveau 0 : six familles de CNN**, entraînées en validation croisée. Chaque fold produit les
probabilités *hors-fold* de ses images de validation, pour que le niveau 1 ne voie jamais la
prédiction d'un réseau sur une image qui a servi à l'entraîner.

| Famille | Architecture | Entrée | Rôle |
|---|---|---|---|
| `resnet192` / `resnet288` | ResNet-18 | 192x128 / 288x192 | 5 classes |
| `sep192` / `sep288` | convolutions séparables + Squeeze-Excitation | 192x128 / 288x192 | 5 classes |
| `spec192` / `spec288` | idem, plus compact | 192x128 / 288x192 | binaire Blockbuster |

**Niveau 1 : `HistGradientBoostingClassifier`** sur les probabilités des CNN et 9 métadonnées de
fichier (dimensions, ratio, poids en octets...), avec :
- pondération des classes en `1/sqrt(freq)` pour le F1 macro ;
- un coefficient unique sur la probabilité Blockbuster, réglé en validation croisée.

Autres choix : dédoublonnage par dhash avant le split, rééquilibrage des classes minoritaires,
augmentation (recadrage, miroir, affine léger, ColorJitter modéré, RandomErasing), mixup,
label smoothing, OneCycleLR, TTA miroir, cache de vignettes et reprise après interruption.
Les justifications chiffrées sont dans les docstrings et commentaires du code.

## Structure

| Fichier | Contenu |
|---|---|
| `data_utils.py` | indexation, dédoublonnage, cache d'images, augmentation, rééquilibrage |
| `features.py` | métadonnées de fichier |
| `model.py` | ResNet-18 écrit à la main |
| `nets.py` | réseau séparable + SE |
| `entrainement.py` | boucle d'entraînement, mixup, TTA, reprise |
| `niveau0.py` | entraîne les 6 familles en validation croisée |
| `niveau1.py` | fusion + génération de `submission.csv` |
| `interpretation.py` | importance par permutation, ablation par bloc, analyse de l'arbitrage |

## Utilisation

Les données ne sont pas incluses dans le dépôt. Placer à la racine :

```
train/              images d'entraînement
test/               images de test
train_labels.csv    labels (nom, classe)
```

Puis :

```bash
pip install -r requirements.txt
python niveau0.py          # long : entraîne les 6 familles (reprise automatique)
python niveau1.py          # écrit submission.csv
python interpretation.py   # analyse du classifieur final
```

La résolution peut être forcée via les variables d'environnement `IMG_H` et `IMG_W`.
Le nombre de folds se règle avec `N_FOLDS` dans `data_utils.py` (5 par défaut).
