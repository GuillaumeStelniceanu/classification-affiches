"""Socle commun aux quatre familles : device, mixup, boucle d'entrainement, TTA, reprise.

Tout ce qui est partage par les reseaux vit ici, pour qu'une famille se resume a
declarer son architecture et sa resolution.

RESOLUTION D'ENTREE. Elle se pilote par les variables d'environnement IMG_H et IMG_W,
lues au niveau MODULE. Ce n'est pas un detail : sous Windows les workers du DataLoader
sont crees par `spawn`, qui reexecute le fichier depuis le debut. Un reglage place dans
une fonction ne les atteindrait pas et ils chargeraient a la mauvaise taille, sans
aucun message d'erreur.

    IMG_H=288 IMG_W=192 python niveau0.py      # bash
    $env:IMG_H=288; $env:IMG_W=192; python niveau0.py       # PowerShell
"""
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import data_utils

if os.environ.get("IMG_H") and os.environ.get("IMG_W"):
    data_utils.IMG_SIZE = (int(os.environ["IMG_H"]), int(os.environ["IMG_W"]))

from data_utils import Posters                                              # noqa: E402

DEV = ("cuda" if torch.cuda.is_available()
       else "mps" if torch.backends.mps.is_available() else "cpu")
AMP = dict(device_type="cuda" if DEV == "cuda" else "cpu", enabled=DEV == "cuda")

# Config commune. width 24 et mixup 0.4 viennent d'une mesure : a width 64 le reseau
# atteignait 0.982 sur son train contre 0.633 en validation, soit 35 points d'ecart.
CFG = dict(epochs=30, batch=128, lr=3e-3, wd=5e-2, mixup=.4, workers=2, seed=42,
           width=24, dropout=.4)
# les paramètre sont à modifier si besoin

def charge(df, train, batch=64, workers=0):
    """DataLoader. Seul l'entrainement utilise des workers : sous Windows chacun est un
    processus qui recharge torch et son contexte CUDA (~680 Mo). Deux jeux de workers
    vivants en meme temps ont deja fait tuer le programme par l'OS."""
    return DataLoader(Posters(df, train=train, with_label="label" in df),
                      batch_size=batch, shuffle=train, drop_last=train,
                      num_workers=workers if train else 0,
                      pin_memory=torch.accelerator.is_available(),
                      persistent_workers=train and workers > 0)


def mixup(x, y, alpha):
    """Melange deux images et deux labels : regularisation cle pour un modele from scratch."""
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


@torch.no_grad()
def probas(model, loader):
    """Probabilites moyennees avec le TTA miroir : une affiche et son symetrique
    horizontal decrivent le meme film, moyenner les deux avis reduit la variance."""
    model.eval()
    out = []
    for x, _ in loader:
        x = x.to(DEV, non_blocking=True)
        p = model(x).softmax(1) + model(torch.flip(x, [3])).softmax(1)
        out.append((p / 2).float().cpu())
    return torch.cat(out).numpy()


def entraine(model, tr, cfg, epochs, avec_mixup=True, tag=""):
    """AdamW + OneCycleLR + mixup + label smoothing + AMP.

    Le checkpoint garde est celui de la DERNIERE epoch, jamais le maximum d'une
    validation : choisir le meilleur parmi 70 evaluations bruitees gonfle le score
    de 2,5 points, c'est mesure. OneCycleLR ramenant le pas d'apprentissage a zero,
    le modele terminal est deja le bon.
    """
    dl = charge(tr, True, cfg["batch"], cfg["workers"])
    crit = nn.CrossEntropyLoss(label_smoothing=.1)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], epochs=epochs,
                                                steps_per_epoch=len(dl))
    scaler = torch.amp.GradScaler(AMP["device_type"], enabled=AMP["enabled"])
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.
        for x, y in dl:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            if avec_mixup:
                xm, ya, yb, lam = mixup(x, y, cfg["mixup"])
                with torch.autocast(**AMP):
                    o = model(xm)
                    loss = lam * crit(o, ya) + (1 - lam) * crit(o, yb)
            else:
                with torch.autocast(**AMP):
                    loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item() * x.size(0)
        if ep % 10 == 0 or ep == epochs:
            print(f"    {tag} epoch {ep:>3}/{epochs} | loss {running / len(dl.dataset):.3f}",
                  flush=True)
    del dl
    return model


def repris_ou_entraine(chemin, fabrique, tr, cfg, epochs, avec_mixup, tag):
    """Recharge le modele s'il existe, sinon l'entraine et le sauvegarde.

    L'ensemble complet demande ~13 h : sans reprise, le moindre incident fait tout
    recommencer. Une limite memoire atteinte au dernier fold a deja coute 2 h 30.
    """
    m = fabrique().to(DEV)
    if chemin.exists():
        m.load_state_dict(torch.load(chemin, map_location=DEV))
        print(f"    {tag} : repris depuis {chemin.name}, pas de reentrainement")
        return m
    chemin.parent.mkdir(parents=True, exist_ok=True)
    m = entraine(m, tr, cfg, epochs, avec_mixup=avec_mixup, tag=tag)
    torch.save(m.state_dict(), chemin)
    return m


def npy_cache(chemin, calcul):
    """Memorise un tableau sur disque : les probabilites ne sont plus perdues au crash."""
    if chemin.exists():
        return np.load(chemin)
    a = calcul()
    chemin.parent.mkdir(parents=True, exist_ok=True)
    np.save(chemin, a)
    return a
