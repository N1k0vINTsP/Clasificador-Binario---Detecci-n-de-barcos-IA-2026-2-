"""Linea base y descriptores clasicos con validacion cruzada estratificada (5 folds)."""
import json
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from datos import RAIZ, cargar_shipsnet
from descriptores import extraer

SEED = 42
OUT = RAIZ / "resultados"
METRICAS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def evaluar(nombre, modelo, F, y, cv, filas, extra=None):
    t0 = time.time()
    r = cross_validate(modelo, F, y, cv=cv, scoring=METRICAS, n_jobs=1, return_estimator=True)
    fila = {"modelo": nombre, "n_feats": F.shape[1]}
    for m in METRICAS:
        fila[m] = r[f"test_{m}"].mean()
        fila[m + "_std"] = r[f"test_{m}"].std()
    # tiempo de inferencia por imagen (solo el clasificador, sin extraer descriptores)
    est = r["estimator"][0]
    t1 = time.perf_counter()
    est.predict(F[:1000])
    fila["ms_img_clf"] = (time.perf_counter() - t1) * 1000 / 1000
    fila["seg_total"] = time.time() - t0
    if extra:
        fila.update(extra)
    if hasattr(est, "best_params_"):
        fila["mejores_params"] = json.dumps([e.best_params_ for e in r["estimator"]])
    filas.append(fila)
    pd.DataFrame(filas).to_csv(OUT / "cv_clasicos.csv", index=False)
    print(f"{nombre:40s} acc={fila['accuracy']:.4f}±{fila['accuracy_std']:.4f} "
          f"f1={fila['f1']:.4f} ({fila['seg_total']:.0f}s)", flush=True)
    return r


def main():
    X, y = cargar_shipsnet()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    filas = []

    t = time.time()
    feats = {n: extraer(X, [n]) for n in ["pixeles", "color", "hog", "lbp"]}
    t_ext = {n: 0.0 for n in feats}
    for n in feats:
        t1 = time.perf_counter()
        extraer(X[:200], [n], n_jobs=1)
        t_ext[n] = (time.perf_counter() - t1) / 200 * 1000
    print("descriptores listos en", round(time.time() - t), "s", {k: v.shape for k, v in feats.items()})

    # 1) linea base: pixeles crudos (40x40x3) sin descriptores
    evaluar("base: pixeles + regresion logistica",
            make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)),
            feats["pixeles"], y, cv, filas, {"ms_desc": t_ext["pixeles"]})
    evaluar("base: pixeles + KNN (k=5)",
            make_pipeline(StandardScaler(), PCA(50, random_state=SEED), KNeighborsClassifier(5)),
            feats["pixeles"], y, cv, filas, {"ms_desc": t_ext["pixeles"]})

    # 2) cada descriptor por separado con el mismo SVM
    for n in ["color", "hog", "lbp"]:
        evaluar(f"{n} + SVM-RBF", make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")),
                feats[n], y, cv, filas, {"ms_desc": t_ext[n]})

    # 3) combinacion de descriptores
    F = np.hstack([feats["hog"], feats["color"], feats["lbp"]])
    ms = t_ext["hog"] + t_ext["color"] + t_ext["lbp"]
    evaluar("hog+color+lbp + SVM-RBF (C=10)",
            make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")), F, y, cv, filas, {"ms_desc": ms})
    evaluar("hog+color+lbp + RandomForest",
            RandomForestClassifier(500, n_jobs=4, random_state=SEED), F, y, cv, filas, {"ms_desc": ms})

    # 4) busqueda de hiperparametros del SVM (CV anidada: 3 folds internos)
    grid = GridSearchCV(
        make_pipeline(StandardScaler(), SVC()),
        {"svc__C": [1, 3, 10, 30, 100], "svc__gamma": ["scale", 1e-4, 3e-4, 1e-3]},
        cv=StratifiedKFold(3, shuffle=True, random_state=SEED), scoring="accuracy", n_jobs=4,
    )
    evaluar("hog+color+lbp + SVM-RBF (grid search)", grid, F, y, cv, filas, {"ms_desc": ms})

    # superficie de sensibilidad C-gamma sobre todo el conjunto (para la figura)
    grid.fit(F, y)
    sens = pd.DataFrame(grid.cv_results_)[["param_svc__C", "param_svc__gamma", "mean_test_score"]]
    sens.to_csv(OUT / "svm_sensibilidad_C_gamma.csv", index=False)


def sensibilidad_hog():
    # un factor a la vez: tamano de celda con 9 orientaciones, y orientaciones con celda de 8 px
    X, y = cargar_shipsnet()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    combos = [(c, 9) for c in (4, 6, 8, 12, 16)] + [(8, o) for o in (6, 12, 18)]
    filas = []
    for celda, orient in combos:
        Fh = extraer(X, ["hog"], hog={"celda": celda, "orient": orient})
        r = cross_validate(make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")),
                           Fh, y, cv=cv, scoring="accuracy", n_jobs=5)
        filas.append({"celda": celda, "orientaciones": orient, "n_feats": Fh.shape[1],
                      "accuracy": r["test_score"].mean(), "std": r["test_score"].std()})
        print("hog", celda, orient, Fh.shape[1], round(r["test_score"].mean(), 4), flush=True)
        pd.DataFrame(filas).to_csv(OUT / "hog_sensibilidad.csv", index=False)


if __name__ == "__main__":
    import sys
    if "hog" in sys.argv[1:]:
        sensibilidad_hog()
    else:
        main()
