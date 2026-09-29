"""Sensibilidad a perturbaciones tipicas de una camara embarcada (desenfoque, ruido, brillo,
resolucion, compresion JPEG, zoom). Se entrena en los folds 1-4 y se mide en el fold 0.

uso: python robustez.py resnet18d_pre cnn_geom
"""
import io
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from datos import RAIZ, cargar_shipsnet
from descriptores import extraer
from entreno import entrenar, predecir
from exp_cnn import CONFIGS
from redes import crear


def perturbar(X, tipo, v):
    out = []
    rng = np.random.default_rng(0)
    for a in X:
        img = Image.fromarray(a)
        if tipo == "desenfoque" and v > 0:
            img = img.filter(ImageFilter.GaussianBlur(v))
        elif tipo == "ruido" and v > 0:
            r = rng.normal(0, v * 255, a.shape)
            img = Image.fromarray(np.clip(a + r, 0, 255).astype(np.uint8))
        elif tipo == "brillo":
            img = ImageEnhance.Brightness(img).enhance(v)
        elif tipo == "resolucion" and v < 1:
            lado = max(8, int(round(80 * v)))
            img = img.resize((lado, lado), Image.BILINEAR).resize((80, 80), Image.BILINEAR)
        elif tipo == "jpeg" and v < 100:
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=int(v))
            img = Image.open(buf).convert("RGB")
        elif tipo == "zoom" and v != 1:
            lado = int(round(80 * v))
            g = img.resize((lado, lado), Image.BILINEAR)
            if v > 1:
                o = (lado - 80) // 2
                img = g.crop((o, o, o + 80, o + 80))
            else:
                fondo = np.asarray(img.resize((1, 1), Image.BOX))[0, 0]
                img = Image.new("RGB", (80, 80), tuple(int(c) for c in fondo))
                img.paste(g, ((80 - lado) // 2, (80 - lado) // 2))
        out.append(np.asarray(img))
    return np.stack(out)


PRUEBAS = {
    "desenfoque": [0, 0.5, 1.0, 1.5, 2.0],
    "ruido": [0, 0.02, 0.05, 0.08, 0.12],
    "brillo": [0.5, 0.7, 1.0, 1.3, 1.6],
    "resolucion": [1.0, 0.75, 0.5, 0.35, 0.25],
    "jpeg": [100, 60, 30, 15, 8],
    "zoom": [0.7, 0.85, 1.0, 1.15, 1.3],
}


def main(configs):
    X, y = cargar_shipsnet()
    tr, va = next(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    modelos = {}
    for n in configs:
        cfg = dict(CONFIGS[n])
        red, kw = cfg.pop("red"), cfg.pop("kw")
        m, _ = entrenar(crear(red, **kw), X[tr], y[tr], semilla=0, log=False, **cfg)
        modelos[n] = (lambda Z, m=m, nm=cfg.get("norm", "global"): predecir(m, Z, nm, tta=True))
        print(n, "entrenado", flush=True)
    nd = ["hog", "color", "lbp"]
    svm = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")).fit(extraer(X[tr], nd), y[tr])
    modelos["hog+color+lbp SVM"] = lambda Z: svm.decision_function(extraer(Z, nd)) > 0

    filas = []
    for tipo, valores in PRUEBAS.items():
        for v in valores:
            Z = perturbar(X[va], tipo, v)
            for n, f in modelos.items():
                p = f(Z)
                acc = float(((np.asarray(p) >= 0.5) == y[va]).mean())
                filas.append({"perturbacion": tipo, "valor": v, "modelo": n, "accuracy": acc})
            print(tipo, v, {r["modelo"]: round(r["accuracy"], 4) for r in filas[-len(modelos):]}, flush=True)
    pd.DataFrame(filas).to_csv(RAIZ / "resultados" / "robustez.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
