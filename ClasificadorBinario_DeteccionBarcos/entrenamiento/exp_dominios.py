"""Efecto de agregar imagenes de alta resolucion (Maxar, otros puertos) al entrenamiento.

Cada fold entrena con: ShipsNet (4/5) + Maxar (4/5, separado por barco para que un mismo barco
no quede en entrenamiento y validacion). Se valida en el 1/5 restante de cada uno y en los
conjuntos externos que nunca se usan para entrenar (escenas Planet y Rotterdam).

uso: python exp_dominios.py resnet18d_pre
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from datos import RAIZ, cargar_shipsnet, ruta_npz
from entreno import entrenar, predecir
from exp_cnn import CONFIGS, externos, metricas

OUT = RAIZ / "resultados"


def correr(nombre):
    cfg = dict(CONFIGS[nombre])
    red, kw = cfg.pop("red"), cfg.pop("kw")
    norm = cfg.get("norm", "global")
    X, y = cargar_shipsnet()
    d = np.load(ruta_npz("ext_maxar.npz"))
    Xm, ym, gm = d["X"], d["y"], d["grupo"]
    ext = externos()
    cv_s = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    cv_m = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(Xm, ym, gm))
    oof_s = np.full(len(y), np.nan)
    oof_m = np.full(len(ym), np.nan)
    p_ext = {n: [] for n in ext}
    filas = []
    for k in range(5):
        (tr, va), (trm, vam) = cv_s[k], cv_m[k]
        t0 = time.time()
        from redes import crear
        m, _ = entrenar(crear(red, **kw), np.concatenate([X[tr], Xm[trm]]), np.concatenate([y[tr], ym[trm]]),
                        semilla=k, log=False, **cfg)
        oof_s[va] = predecir(m, X[va], norm, tta=True)
        oof_m[vam] = predecir(m, Xm[vam], norm, tta=True)
        f = {"config": nombre + "+maxar", "fold": k, "seg_entreno": time.time() - t0}
        f.update({"shipsnet_" + a: b for a, b in metricas(y[va], oof_s[va]).items()})
        f.update({"maxar_" + a: b for a, b in metricas(ym[vam], oof_m[vam]).items()})
        for n, (Xe, ye) in ext.items():
            pe = predecir(m, Xe, norm, tta=True)
            p_ext[n].append(pe)
            f.update({f"{n}_{a}": b for a, b in metricas(ye, pe).items()})
        filas.append(f)
        print(f"fold {k}: shipsnet={f['shipsnet_accuracy']:.4f} maxar={f['maxar_accuracy']:.4f} "
              + " ".join(f"{n}={f[n + '_accuracy']:.4f}" for n in ext) + f" ({f['seg_entreno'] / 60:.1f} min)", flush=True)
    np.savez(OUT / "oof" / f"{nombre}+maxar.npz", p_shipsnet=oof_s, y=y, p_maxar=oof_m, y_maxar=ym,
             **{n: np.array(v) for n, v in p_ext.items()})
    df = pd.DataFrame(filas)
    ruta = OUT / "cv_dominios_folds.csv"
    df.to_csv(ruta, mode="a", header=not ruta.exists(), index=False)
    cols = [c for c in df.columns if c.endswith(("accuracy", "recall", "precision"))]
    print(nombre + "+maxar", json.dumps({c: round(df[c].mean(), 4) for c in cols}), flush=True)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        correr(n)
