"""Genera las figuras del informe a partir de los CSV de resultados."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datos import RAIZ

RES = RAIZ / "resultados"
FIG = RAIZ / "docs" / "figuras"
FIG.mkdir(parents=True, exist_ok=True)

AZUL, NARANJA, VERDE, AMARILLO, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
TINTA, TINTA2 = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
    "axes.edgecolor": "#d6d5d0", "axes.linewidth": 0.8, "axes.labelcolor": TINTA2,
    "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.grid": True,
    "grid.color": "#ebeae6", "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 10, "axes.titleweight": "bold",
})


def guardar(fig, nombre):
    fig.savefig(FIG / nombre)
    plt.close(fig)
    print("->", nombre)


def comparacion_modelos():
    filas = []
    c = pd.read_csv(RES / "cv_clasicos.csv")
    for _, r in c.iterrows():
        filas.append((r["modelo"], r["accuracy"], r["accuracy_std"]))
    if (RES / "cv_cnn_folds.csv").exists():
        d = pd.read_csv(RES / "cv_cnn_folds.csv").drop_duplicates(["config", "fold"], keep="last")
        nombres = {"cnn_sin_aug": "CNN propia, sin aumento", "cnn_geom": "CNN propia + rotaciones/espejos",
                   "cnn_aug": "CNN propia + aumento completo", "resnet18d_scratch": "ResNet18-D desde cero",
                   "resnet18d_pre": "ResNet18-D ImageNet (ajuste fino)", "mnv3_pre": "MobileNetV3 ImageNet",
                   "convnext_atto_pre": "ConvNeXt-Atto ImageNet"}
        for cfg, g in d.groupby("config"):
            if len(g) == 5:
                filas.append((nombres.get(cfg, cfg) + " + TTA", g["accuracy_tta"].mean(), g["accuracy_tta"].std(ddof=0)))
    df = pd.DataFrame(filas, columns=["modelo", "acc", "std"]).sort_values("acc")
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(df) + 0.8))
    colores = [AZUL if "ImageNet" in m else (VERDE if "CNN" in m or "ResNet" in m else GRIS) for m in df.modelo]
    ax.barh(df.modelo, df.acc * 100, xerr=df["std"] * 100, color=colores, height=0.6,
            error_kw=dict(ecolor=TINTA2, lw=0.8, capsize=2))
    for i, (a, s) in enumerate(zip(df.acc, df["std"])):
        ax.text(a * 100 + s * 100 + 0.15, i, f"{a * 100:.2f} %", va="center", fontsize=8, color=TINTA)
    ax.axvline(98, color=NARANJA, lw=1)
    ax.text(98, len(df) - 0.4, " meta 98 %", color=TINTA2, fontsize=8, va="bottom")
    ax.set_xlim(75, 101.5)
    ax.set_xlabel("accuracy en validación cruzada estratificada, 5 folds (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("De la línea base a la red final (ShipsNet, 4000 imágenes)", loc="left")
    guardar(fig, "comparacion_modelos.png")
    df.sort_values("acc", ascending=False).to_csv(RES / "tabla_comparativa.csv", index=False)


def sensibilidad_svm():
    s = pd.read_csv(RES / "svm_sensibilidad_C_gamma.csv")
    s["gamma"] = s["param_svc__gamma"].astype(str)
    fig, ax = plt.subplots(figsize=(5.2, 3))
    for (g, sub), col in zip(s.groupby("gamma", sort=False), [AZUL, NARANJA, VERDE, AMARILLO]):
        sub = sub.sort_values("param_svc__C")
        # scale y 3e-4 dan casi lo mismo; scale va con marcador hueco para que se vean las dos
        estilo = dict(mfc="white", mew=1.5, ms=7, zorder=3) if g == "scale" else dict(ms=4)
        ax.plot(sub["param_svc__C"], sub["mean_test_score"] * 100, "-o", color=col, lw=2, label=f"γ = {g}", **estilo)
    ax.set_xscale("log")
    ax.set_xlabel("C (escala log)")
    ax.set_ylabel("accuracy CV interna (%)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("SVM-RBF sobre HOG+color+LBP: sensibilidad a C y γ", loc="left")
    guardar(fig, "sensibilidad_svm.png")


def sensibilidad_hog():
    ruta = RES / "hog_sensibilidad.csv"
    if not ruta.exists():
        return
    h = pd.read_csv(ruta)
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)
    a = h[h.orientaciones == 9].sort_values("celda")
    axs[0].errorbar(a.celda, a.accuracy * 100, yerr=a["std"] * 100, fmt="-o", color=AZUL, lw=2, ms=4, capsize=2)
    axs[0].set_xlabel("tamaño de celda HOG (px), 9 orientaciones")
    axs[0].set_ylabel("accuracy CV (%)")
    b = h[h.celda == 8].sort_values("orientaciones")
    axs[1].errorbar(b.orientaciones, b.accuracy * 100, yerr=b["std"] * 100, fmt="-o", color=AZUL, lw=2, ms=4, capsize=2)
    axs[1].set_xlabel("orientaciones, celda de 8 px")
    fig.suptitle("Sensibilidad del descriptor HOG (SVM-RBF, C=10)", x=0.02, ha="left", fontweight="bold", fontsize=10)
    guardar(fig, "sensibilidad_hog.png")


def dominios():
    ruta = RES / "cv_dominios_folds.csv"
    if not ruta.exists():
        return
    d = pd.read_csv(ruta).drop_duplicates(["config", "fold"], keep="last")
    c = pd.read_csv(RES / "cv_cnn_folds.csv").drop_duplicates(["config", "fold"], keep="last")
    filas = []
    for cfg in ["resnet18d_pre", "cnn_geom"]:
        s = c[c.config == cfg]
        m = d[d.config == cfg + "+maxar"]
        if len(s) == 5:
            filas.append((cfg, "solo ShipsNet", s.accuracy_tta.mean(), s.ext_planet_acc.mean(), s.ext_rotterdam_acc.mean(), s.ext_rotterdam_recall.mean()))
        if len(m) == 5:
            filas.append((cfg, "ShipsNet + Maxar", m.shipsnet_accuracy.mean(), m.ext_planet_accuracy.mean(), m.ext_rotterdam_accuracy.mean(), m.ext_rotterdam_recall.mean()))
    t = pd.DataFrame(filas, columns=["red", "entrenamiento", "cv_shipsnet", "planet_ext", "rotterdam_acc", "rotterdam_recall"])
    t.to_csv(RES / "tabla_dominios.csv", index=False)
    t = t[t.red == "resnet18d_pre"]
    if t.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    cats = ["CV ShipsNet", "Escenas Planet\n(externo)", "Rotterdam\naccuracy", "Rotterdam\nrecall barcos"]
    x = np.arange(len(cats))
    w = 0.36
    for j, (lab, col) in enumerate([("solo ShipsNet", GRIS), ("ShipsNet + Maxar", AZUL)]):
        r = t[t.entrenamiento == lab]
        if r.empty:
            continue
        v = r.iloc[0][["cv_shipsnet", "planet_ext", "rotterdam_acc", "rotterdam_recall"]].values.astype(float) * 100
        ax.bar(x + (j - 0.5) * w, v, w * 0.92, color=col, label=lab)
        for xi, vi in zip(x + (j - 0.5) * w, v):
            ax.text(xi, vi + 1, f"{vi:.1f}", ha="center", fontsize=7.5, color=TINTA)
    ax.set_xticks(x, cats)
    ax.set_ylim(0, 122)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("%")
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    ax.grid(axis="x", visible=False)
    ax.set_title("ResNet18-D: entrenar solo con ShipsNet vs. agregar puertos Maxar", loc="left")
    guardar(fig, "dominios.png")


def robustez():
    ruta = RES / "robustez.csv"
    if not ruta.exists():
        return
    r = pd.read_csv(ruta)
    tipos = list(r.perturbacion.unique())
    fig, axs = plt.subplots(1, len(tipos), figsize=(2.1 * len(tipos), 2.4), sharey=True)
    modelos = list(r.modelo.unique())
    colores = [AZUL, VERDE, NARANJA, AMARILLO]
    for ax, t in zip(axs, tipos):
        s = r[r.perturbacion == t]
        for mdl, col in zip(modelos, colores):
            q = s[s.modelo == mdl]
            ax.plot(range(len(q)), q.accuracy * 100, "-o", color=col, lw=2, ms=3.5, label=mdl)
        ax.set_xticks(range(len(q)), [str(v) for v in q.valor], fontsize=7)
        ax.set_title(t, fontsize=9)
    axs[0].set_ylabel("accuracy fold 0 (%)")
    axs[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle("Robustez ante degradaciones típicas de la cámara del UAV", x=0.02, ha="left", fontweight="bold", fontsize=10)
    guardar(fig, "robustez.png")


def matriz_oof():
    info = json.loads((RAIZ / "modelo" / "info.json").read_text(encoding="utf-8"))
    cfgs = [c["config"] for c in info.get("configuraciones", [])]
    if not cfgs:
        return
    ps = [np.load(RES / "oof" / f"{c}.npz") for c in cfgs if (RES / "oof" / f"{c}.npz").exists()]
    if not ps:
        return
    y = ps[0]["y"]
    p = np.mean([d["p_tta"] for d in ps], axis=0)
    yp = (p >= 0.5).astype(int)
    cm = np.array([[((y == 1) & (yp == 1)).sum(), ((y == 1) & (yp == 0)).sum()],
                   [((y == 0) & (yp == 1)).sum(), ((y == 0) & (yp == 0)).sum()]])
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else TINTA)
    ax.set_xticks([0, 1], ["pred. barco", "pred. no barco"])
    ax.set_yticks([0, 1], ["barco", "no barco"])
    ax.grid(False)
    ax.set_title("Matriz de confusión, predicciones fuera de fold", loc="left", fontsize=9)
    guardar(fig, "matriz_cv.png")


if __name__ == "__main__":
    comparacion_modelos()
    sensibilidad_svm()
    sensibilidad_hog()
    dominios()
    robustez()
    if (RAIZ / "modelo" / "info.json").exists():
        matriz_oof()
