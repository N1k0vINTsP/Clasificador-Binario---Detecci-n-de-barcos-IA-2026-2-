"""Validacion cruzada (5 folds, mismos cortes que exp_clasicos) para las redes.

uso: python exp_cnn.py cnn_aug resnet18d_pre ...   (ver CONFIGS)
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from datos import RAIZ, cargar_npz, cargar_shipsnet, ruta_npz
from entreno import entrenar, predecir
from redes import crear, n_params

OUT = RAIZ / "resultados"
SEED = 42

CONFIGS = {
    # red propia, sin y con aumento de datos
    "cnn_sin_aug": dict(red="cnn", kw=dict(ancho=32), aug="ninguno", epocas=25, lr=2e-3),
    "cnn_geom": dict(red="cnn", kw=dict(ancho=32), aug="geometrico", epocas=25, lr=2e-3),
    "cnn_aug": dict(red="cnn", kw=dict(ancho=32), aug="completo", epocas=25, lr=2e-3),
    "cnn_aug_normimg": dict(red="cnn", kw=dict(ancho=32), aug="completo", epocas=25, lr=2e-3, norm="imagen"),
    "cnn16_aug": dict(red="cnn", kw=dict(ancho=16), aug="completo", epocas=25, lr=2e-3),
    # transferencia desde ImageNet
    "resnet18d_pre": dict(red="resnet18d", kw=dict(), aug="completo", epocas=12, lr=1e-3, wd=1e-4),
    "resnet18d_scratch": dict(red="resnet18d", kw=dict(preentrenado=False), aug="completo", epocas=12, lr=2e-3),
    "convnext_atto_pre": dict(red="convnext_atto", kw=dict(), aug="completo", epocas=12, lr=1e-3, wd=5e-2),
    "effb0_pre": dict(red="efficientnet_b0", kw=dict(), aug="completo", epocas=12, lr=1e-3, wd=1e-4),
    "mnv3_pre": dict(red="mobilenetv3_large_100", kw=dict(), aug="completo", epocas=12, lr=1e-3, wd=1e-4),
}


def metricas(y, p, umbral=0.5):
    yp = (p >= umbral).astype(int)
    return dict(accuracy=accuracy_score(y, yp), precision=precision_score(y, yp, zero_division=0),
                recall=recall_score(y, yp), f1=f1_score(y, yp), roc_auc=roc_auc_score(y, p))


def externos():
    # conjuntos armados a mano fuera de ShipsNet (ver armar_externos.py)
    out = {}
    for n in ("ext_planet", "ext_rotterdam"):
        if ruta_npz(f"{n}.npz"):
            out[n] = cargar_npz(f"{n}.npz")
    return out


def correr(nombre, folds=range(5)):
    cfg = dict(CONFIGS[nombre])
    red, kw = cfg.pop("red"), cfg.pop("kw")
    X, y = cargar_shipsnet()
    ext = externos()
    p_ext = {n: [] for n in ext}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    oof_tta = np.full(len(y), np.nan)
    filas = []
    for k, (tr, va) in enumerate(cv.split(X, y)):
        if k not in folds:
            continue
        t0 = time.time()
        m = crear(red, **kw)
        m, hist = entrenar(m, X[tr], y[tr], semilla=k, log=False, **cfg)
        t_ent = time.time() - t0
        norm = cfg.get("norm", "global")
        oof[va] = predecir(m, X[va], norm)
        oof_tta[va] = predecir(m, X[va], norm, tta=True)
        f = {"config": nombre, "fold": k, "params": n_params(m), "seg_entreno": t_ent}
        f.update({m_: v for m_, v in metricas(y[va], oof[va]).items()})
        f.update({m_ + "_tta": v for m_, v in metricas(y[va], oof_tta[va]).items()})
        for n, (Xe, ye) in ext.items():
            pe = predecir(m, Xe, norm, tta=True)
            p_ext[n].append(pe)
            f[n + "_acc"] = float(((pe >= 0.5) == ye).mean())
            f[n + "_recall"] = float((pe[ye == 1] >= 0.5).mean())
            f[n + "_espec"] = float((pe[ye == 0] < 0.5).mean())
        filas.append(f)
        print(f"{nombre} fold {k}: acc={f['accuracy']:.4f} acc_tta={f['accuracy_tta']:.4f} "
              f"({t_ent/60:.1f} min)", flush=True)
    (OUT / "oof").mkdir(exist_ok=True)
    np.savez(OUT / "oof" / f"{nombre}.npz", p=oof, p_tta=oof_tta, y=y,
             **{n: np.array(v) for n, v in p_ext.items()})
    df = pd.DataFrame(filas)
    ruta = OUT / "cv_cnn_folds.csv"
    df.to_csv(ruta, mode="a", header=not ruta.exists(), index=False)
    cols = ["accuracy", "accuracy_tta", "f1_tta"] + [c for c in df.columns if c.startswith("ext_")]
    print(nombre, json.dumps({c: round(df[c].mean(), 4) for c in cols}), flush=True)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        correr(n)
