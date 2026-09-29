"""Imprime en markdown las tablas del informe a partir de los CSV de resultados."""
import json

import numpy as np
import pandas as pd

from datos import RAIZ

RES = RAIZ / "resultados"


def pct(x, d=2):
    return f"{100 * x:.{d}f}"


def clasicos():
    c = pd.read_csv(RES / "cv_clasicos.csv")
    print("| Modelo | Descriptores | Accuracy (%) | Precisión (%) | Recall (%) | F1 | AUC |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in c.iterrows():
        print(f"| {r.modelo} | {int(r.n_feats)} | {pct(r.accuracy)} ± {pct(r.accuracy_std)} | {pct(r.precision)} | "
              f"{pct(r.recall)} | {r.f1:.3f} | {r.roc_auc:.4f} |")
    if "mejores_params" in c and c.mejores_params.notna().any():
        print("\nparametros elegidos por fold:", c.mejores_params.dropna().iloc[0])


def redes():
    d = pd.read_csv(RES / "cv_cnn_folds.csv").drop_duplicates(["config", "fold"], keep="last")
    nombres = {"cnn_sin_aug": "CNN propia, sin aumento", "cnn_geom": "CNN propia, 8 orientaciones",
               "cnn_aug": "CNN propia, aumento completo", "resnet18d_scratch": "ResNet18-D desde cero",
               "resnet18d_pre": "ResNet18-D ImageNet", "mnv3_pre": "MobileNetV3-L ImageNet",
               "convnext_atto_pre": "ConvNeXt-Atto ImageNet", "effb0_pre": "EfficientNet-B0 ImageNet"}
    print("| Red | Parámetros (M) | Accuracy (%) | Accuracy con TTA (%) | F1 con TTA | Escenas Planet (%) | Rotterdam (%) | Recall Rotterdam (%) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for cfg, g in d.groupby("config", sort=False):
        if len(g) < 5:
            continue
        print(f"| {nombres.get(cfg, cfg)} | {g.params.iloc[0] / 1e6:.2f} | {pct(g.accuracy.mean())} ± {pct(g.accuracy.std(ddof=0))} | "
              f"{pct(g.accuracy_tta.mean())} ± {pct(g.accuracy_tta.std(ddof=0))} | {g.f1_tta.mean():.3f} | "
              f"{pct(g.ext_planet_acc.mean(), 1)} | {pct(g.ext_rotterdam_acc.mean(), 1)} | {pct(g.ext_rotterdam_recall.mean(), 1)} |")


def dominios():
    t = pd.read_csv(RES / "tabla_dominios.csv")
    print("| Red | Entrenamiento | CV ShipsNet (%) | Escenas Planet (%) | Rotterdam accuracy (%) | Rotterdam recall (%) |")
    print("|---|---|---:|---:|---:|---:|")
    for _, r in t.iterrows():
        print(f"| {r.red} | {r.entrenamiento} | {pct(r.cv_shipsnet)} | {pct(r.planet_ext, 1)} | {pct(r.rotterdam_acc, 1)} | {pct(r.rotterdam_recall, 1)} |")
    d = pd.read_csv(RES / "cv_dominios_folds.csv").drop_duplicates(["config", "fold"], keep="last")
    for cfg, g in d.groupby("config"):
        print(f"\n{cfg}: maxar acc {pct(g.maxar_accuracy.mean())} ± {pct(g.maxar_accuracy.std(ddof=0))}, "
              f"precision {pct(g.maxar_precision.mean())}, recall {pct(g.maxar_recall.mean())}; "
              f"rotterdam precision {pct(g.ext_rotterdam_precision.mean())}")


def robustez():
    r = pd.read_csv(RES / "robustez.csv")
    tabla = r.pivot_table(index=["perturbacion", "valor"], columns="modelo", values="accuracy", sort=False)
    print("| Perturbación | Nivel | " + " | ".join(tabla.columns) + " |")
    print("|---|---:|" + "---:|" * len(tabla.columns))
    for (p, v), fila in tabla.iterrows():
        print(f"| {p} | {v} | " + " | ".join(pct(x, 1) for x in fila.values) + " |")


def latencia():
    lat = pd.read_csv(RES / "latencia.csv")
    print("| Red | Parámetros (M) | ONNX (MB) | ms por imagen | ms con TTA x8 |")
    print("|---|---:|---:|---:|---:|")
    for _, r in lat.iterrows():
        print(f"| {r.modelo} | {r.parametros_M:.2f} | {r.onnx_MB:.1f} | {r.ms_1_imagen:.2f} | {r.ms_tta_8:.2f} |")


if __name__ == "__main__":
    for f in (clasicos, redes, dominios, robustez, latencia):
        print(f"\n### {f.__name__}\n")
        try:
            f()
        except FileNotFoundError as e:
            print("(falta)", e.filename)
    info = json.loads((RAIZ / "modelo" / "info.json").read_text(encoding="utf-8"))
    print("\ninfo.json:", json.dumps({k: info[k] for k in ("cv", "prueba_externa", "rendimiento_onnx") if k in info}, indent=1))
