"""Indexation, dédoublonnage, cache d'images, augmentation et rééquilibrage."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

IMG_DIR = Path("train")
TEST_DIR = Path("test")
LABELS_CSV = Path("train_labels.csv")

CLASSES = ["Animation", "Blockbuster", "Horreur", "Comedie", "Art&Essai"]
IMG_SIZE = (192, 128)  # (H, W). NE PAS AUGMENTER. Le 288x192 a ete teste : il gagne
                       # +2.2 points en hors-fold (0.6804 -> 0.7024) et en PERD 1.5 sur
                       # Kaggle (0.66938 -> 0.65475). Plus de pixels = plus de detail
                       # specifique au train, qui ne transfere pas. Doit rester <= CACHE_SIZE.
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
EXTS = {".png", ".jpg", ".jpeg", ".webp"}
HASH_MODE = "dhash"    # "dhash" = robuste au ré-encodage/redimensionnement ; "exact" = pixels stricts
DUP_DIST = 0           # tolérance en bits. Mesuré : à 4, le dhash regroupe à tort des
                       # affiches sombres pourtant très différentes (écarts pixel jusqu'à
                       # 0.41) et 13 images étaient supprimées à tort. À 0, les 3 seules
                       # suppressions sont de vrais doublons et aucun conflit de label ne subsiste.
N_FOLDS = 5          # validation croisee du niveau 0 ; niveau1 doit lire le meme nombre de folds
BALANCE_CAP = .5       # effectif cible des classes minoritaires, en fraction de la majoritaire
AFFINE = dict(degrees=7, translate=(.05, .05))   # rotation/translation légères :
                       # régularisation géométrique, qui remplace celle retirée au
                       # ColorJitter sans toucher aux statistiques de couleur.
JITTER = (.12, .12, .08, .02)   # (luminosité, contraste, saturation, teinte).
                       # Mesuré : la luminosité moyenne va de 0.302 (Horreur) à 0.500 (Comedie),
                       # soit 0.20 d'amplitude. L'ancien réglage .3 tirait un facteur dans
                       # [0.7, 1.3], donc perturbait PLUS que l'écart entre classes : on
                       # apprenait au modèle à ignorer son indice le plus discriminant.

CACHE_DIR = Path("cache")
CACHE_SIZE = (256, 384)   # (W, H). Fige : reste valable meme si IMG_SIZE change, tant
                          # qu'il ne le depasse pas. Evite de reconstruire le cache.


def visual_hash(path, mode=HASH_MODE, size=8):
    """Empreinte du contenu de l'image, indépendante du nom de fichier.

    dhash : gradients horizontaux sur une vignette 9x8 en niveaux de gris -> 64 bits.
            Deux copies de la même affiche donnent la même empreinte même si l'une a été
            ré-encodée en JPEG ou redimensionnée.
    exact : MD5 des pixels bruts, ne rapproche que des copies rigoureusement identiques.
    """
    img = Image.open(path)
    if mode == "exact":
        return hashlib.md5(img.convert("RGB").tobytes()).hexdigest()
    px = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    return np.packbits(px[:, 1:] > px[:, :-1]).tobytes().hex()


def load_index(img_dir=IMG_DIR, labels_csv=LABELS_CSV):
    """DataFrame [name, label, path]. Le CSV cite des .jpg, le disque contient des .png :
    l'appariement se fait sur le nom sans extension."""
    files = {p.stem: p for p in Path(img_dir).iterdir() if p.suffix.lower() in EXTS}
    df = pd.read_csv(labels_csv, header=None, names=["name", "label"], dtype=str)
    df = df[df["label"].str.strip().str.isdigit()]  # retire l'en-tête éventuel
    df["label"] = df["label"].astype(int)
    df["path"] = df["name"].map(lambda n: files.get(Path(n.strip()).stem))
    missing = df["path"].isna().sum()
    if missing:
        print(f"[!] {missing} lignes du CSV sans image sur le disque -> ignorées")
    return df.dropna(subset=["path"]).reset_index(drop=True)


def cluster_hashes(hexes, max_dist=DUP_DIST):
    """Regroupe les empreintes distantes de <= max_dist bits (glouton, déterministe).
    Tolérer quelques bits d'écart rattrape les copies ré-encodées en JPEG, que
    l'égalité stricte laisserait passer."""
    raw = np.frombuffer(b"".join(bytes.fromhex(h) for h in hexes), dtype=np.uint8)
    bits = np.unpackbits(raw.reshape(len(hexes), -1), axis=1)
    grp = np.full(len(bits), -1)
    for i in range(len(bits)):
        if grp[i] < 0:
            grp[((bits[i] != bits).sum(1) <= max_dist) & (grp < 0)] = i
    return grp


def deduplicate(df):
    """Retire les images identiques (noms différents, même contenu) AVANT le split,
    seul moyen d'éviter qu'une même affiche se retrouve en train et en test."""
    df = df.assign(g=cluster_hashes(df["path"].map(visual_hash).tolist()))
    conflicts = int(df.groupby("g")["label"].nunique().gt(1).sum())
    n_dup = int(df.duplicated("g").sum())
    df = df.drop_duplicates("g").drop(columns="g").reset_index(drop=True)
    print(f"Dédoublonnage : {n_dup} doublons retirés, {len(df)} images conservées "
          f"({conflicts} groupes à labels contradictoires -> si ce nombre est élevé, "
          f"baisser DUP_DIST vers 0)")
    return df


def test_frame(test_dir=TEST_DIR, names_csv=None):
    """DataFrame [name, path] des images de /test, dans l'ordre de rendu."""
    files = {f.stem: f for f in Path(test_dir).iterdir() if f.suffix.lower() in EXTS}
    if names_csv:  # on respecte strictement l'ordre du fichier fourni par l'enseignant
        names = pd.read_csv(names_csv, header=None)[0].astype(str).str.strip().tolist()
    else:
        names = sorted(f"{s}.jpg" for s in files)
    absents = [n for n in names if Path(n).stem not in files]
    if absents:
        raise FileNotFoundError(f"{len(absents)} images absentes de {test_dir} : {absents[:3]}")
    return pd.DataFrame({"name": names, "path": [files[Path(n).stem] for n in names]})


def make_tf(train):
    if not train:
        return transforms.Compose([transforms.Resize(IMG_SIZE),
                                   transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(.65, 1.), ratio=(.6, .78)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(**AFFINE),
        transforms.ColorJitter(*JITTER),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=.35),
    ])


class Posters(Dataset):
    def __init__(self, df, train, with_label=True):
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist() if with_label else [-1] * len(df)
        self.tf = make_tf(train)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), self.labels[i]


def cache_path(src, size=CACHE_SIZE):
    """Chemin de la vignette correspondant à une image source."""
    src = Path(src)
    return CACHE_DIR / f"{size[0]}x{size[1]}" / src.parent.name / f"{src.stem}.jpg"


def use_cache(df, size=CACHE_SIZE, verbose=True):
    """Remplace les chemins par ceux d'un cache pré-redimensionné, construit au besoin.

    Mesuré : décoder un JPEG d'origine (500x750 en médiane) coûte 5.6 ms et l'augmenter
    7.5 ms, soit 13.1 ms par image. Depuis une vignette 256x384 le total tombe vers 6 ms.
    Le GPU n'était utilisé qu'à 6 % : c'est le décodage qui limitait l'entraînement.

    Le cache est écrit une seule fois ; les runs suivants le réutilisent.
    """
    cibles = [cache_path(p, size) for p in df["path"]]
    manquants = [(s, c) for s, c in zip(df["path"], cibles) if not c.exists()]
    if manquants:
        if verbose:
            print(f"Construction du cache {size[0]}x{size[1]} : {len(manquants)} images...")
        for src, dst in manquants:
            dst.parent.mkdir(parents=True, exist_ok=True)
            Image.open(src).convert("RGB").resize(size, Image.BICUBIC).save(dst, quality=92)
        if verbose:
            print(f"  -> {CACHE_DIR / f'{size[0]}x{size[1]}'}")
    return df.assign(path=cibles)


def oversample_minorities(train_df, cap=BALANCE_CAP, seed=42, verbose=True):
    """Complète les classes minoritaires jusqu'à `cap` x l'effectif de la majoritaire,
    en retirant au sort des images de la classe parmi celles déjà présentes.

    Ces copies ne sont pas des doublons figés : make_tf(train=True) applique à CHAQUE
    lecture un recadrage, un miroir horizontal et une variation colorimétrique tirés au
    hasard. Une même affiche dupliquée 7 fois donne donc 7 images différentes dans
    l'epoch, et 7 autres à l'epoch suivante. Écrire les variantes sur le disque figerait
    au contraire une seule version par copie, donc moins de diversité, pas plus.

    Seul le split d'apprentissage est concerné : la validation garde les proportions
    réelles, sinon son score cesserait d'être représentatif.
    """
    counts = train_df["label"].value_counts()
    cible = int(counts.max() * cap)
    rng = np.random.default_rng(seed)
    parts = [train_df]
    for label, g in train_df.groupby("label"):
        manque = cible - len(g)
        if manque > 0:
            parts.append(g.iloc[rng.integers(0, len(g), manque)])
    out = pd.concat(parts, ignore_index=True)
    if verbose:
        apres = out["label"].value_counts()
        print(f"Rééquilibrage (cible {cible}/classe) : {len(train_df)} -> {len(out)} tirages")
        for label in sorted(counts.index):
            ajout = apres[label] - counts[label]
            nom = CLASSES[label] if label < len(CLASSES) else str(label)
            print(f"  {label} {nom:<12} {counts[label]:>5} -> {apres[label]:>5}  (+{ajout})")
    return out
