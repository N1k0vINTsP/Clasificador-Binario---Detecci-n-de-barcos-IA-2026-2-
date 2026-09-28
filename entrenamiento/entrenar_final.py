"""Entrena el modelo que usa la interfaz y lo exporta a ONNX.

Se entrena con todo lo disponible: ShipsNet + puertos Maxar + Rotterdam. Las metricas de
validacion cruzada y de prueba externa que se guardan en info.json vienen de exp_dominios.py
(modelos que no vieron esos datos), no de este entrenamiento.

uso: python entrenar_final.py resnet18d_pre [otra_config ...]
"""
import json
import sys
import time

import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from datos import RAIZ, cargar_npz, cargar_shipsnet, ruta_npz
from entreno import entrenar
from exp_cnn import CONFIGS
from exportar import exportar, guardar_info
from redes import crear, n_params

OUT = RAIZ / "resultados"


def resumen_cv(nombres):
    # predicciones fuera de fold de exp_dominios (ShipsNet + Maxar, sin Rotterdam)
    ds = [np.load(OUT / "oof" / f"{n}+maxar.npz") for n in nombres]
    y, ym = ds[0]["y"], ds[0]["y_maxar"]
    p = np.mean([d["p_shipsnet"] for d in ds], axis=0)
    pm = np.mean([d["p_maxar"] for d in ds], axis=0)
    yp, ypm = (p >= 0.5).astype(int), (pm >= 0.5).astype(int)
    accs = [float((yp[va] == y[va]).mean()) for _, va in StratifiedKFold(5, shuffle=True, random_state=42).split(y, y)]
    cv = {"accuracy": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
          "precision": float(precision_score(y, yp)), "recall": float(recall_score(y, yp)),
          "errores": int((yp != y).sum()), "n": int(len(y)),
          "maxar_accuracy": float((ypm == ym).mean()), "maxar_n": int(len(ym))}
    ext = {}
    for nom in ("ext_planet", "ext_rotterdam"):
        if nom in ds[0]:
            ye = cargar_npz(f"{nom}.npz")[1]
            pe = np.mean([d[nom].mean(0) for d in ds], axis=0)
            ype = (pe >= 0.5).astype(int)
            ext[nom] = {"accuracy": float((ype == ye).mean()), "precision": float(precision_score(ye, ype)),
                        "recall": float(recall_score(ye, ype)), "n": int(len(ye))}
    return cv, ext


def actualizar_info(nombres):
    # recalcula las metricas de info.json sin reentrenar (por si cambian los resultados de exp_dominios)
    ruta = RAIZ / "modelo" / "info.json"
    info = json.loads(ruta.read_text(encoding="utf-8"))
    info["cv"], info["prueba_externa"] = resumen_cv(nombres)
    ruta.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"cv": info["cv"], "externos": info["prueba_externa"]}, indent=2))


def main(nombres):
    X, y = cargar_shipsnet()
    partes = [(X, y)]
    for n in ("ext_maxar.npz", "ext_rotterdam.npz"):
        if ruta_npz(n):
            partes.append(cargar_npz(n))
    Xt = np.concatenate([p[0] for p in partes])
    yt = np.concatenate([p[1] for p in partes])
    print("entrenamiento:", len(yt), "imagenes,", int(yt.sum()), "con barco", flush=True)

    modelos, normas, detalle = [], [], []
    for k, n in enumerate(nombres):
        cfg = dict(CONFIGS[n])
        red, kw = cfg.pop("red"), cfg.pop("kw")
        t0 = time.time()
        m, _ = entrenar(crear(red, **kw), Xt, yt, semilla=100 + k, log=True, **cfg)
        modelos.append(m)
        normas.append(cfg.get("norm", "global"))
        detalle.append({"config": n, "red": red, "params": n_params(m), "min_entreno": round((time.time() - t0) / 60, 1)})

    carpeta = RAIZ / "modelo"
    carpeta.mkdir(exist_ok=True)
    rendimiento = exportar(modelos, normas, carpeta / "barcos.onnx")
    cv, ext = resumen_cv(nombres)
    desc = " + ".join(d["red"] for d in detalle) + " preentrenada en ImageNet, ajuste fino, TTA x8"
    guardar_info(carpeta, "barcos.onnx", {
        "descripcion": desc,
        "configuraciones": detalle,
        "datos_entrenamiento": {"imagenes": int(len(yt)), "barcos": int(yt.sum()),
                                "fuentes": ["ShipsNet (Kaggle)", "Maxar Open Data, 7 puertos", "SpaceNet 6 Rotterdam"]},
        "cv": cv,
        "prueba_externa": ext,
        "rendimiento_onnx": rendimiento,
        "fecha": time.strftime("%Y-%m-%d %H:%M"),
    })
    print(json.dumps({"cv": cv, "externos": ext, "onnx": rendimiento}, indent=2))


if __name__ == "__main__":
    if sys.argv[1] == "--info":
        actualizar_info(sys.argv[2:])
    else:
        main(sys.argv[1:])
