"""Repite la mejor combinacion clasica con la celda HOG que salio mejor en la sensibilidad (16 px)."""
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import numpy as np

from datos import RAIZ, cargar_shipsnet
from descriptores import extraer
from exp_clasicos import METRICAS, SEED

X, y = cargar_shipsnet()
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
F = np.hstack([extraer(X, ["hog"], hog={"celda": 16}), extraer(X, ["color"]), extraer(X, ["lbp"])])
filas = []
for nombre, modelo in [
    ("hog16+color+lbp + SVM-RBF (C=10)", make_pipeline(StandardScaler(), SVC(C=10, gamma="scale"))),
    ("hog16+color+lbp + SVM-RBF (grid search)", GridSearchCV(
        make_pipeline(StandardScaler(), SVC()),
        {"svc__C": [1, 3, 10, 30, 100], "svc__gamma": ["scale", 3e-4, 1e-3, 3e-3]},
        cv=StratifiedKFold(3, shuffle=True, random_state=SEED), scoring="accuracy", n_jobs=2)),
]:
    r = cross_validate(modelo, F, y, cv=cv, scoring=METRICAS, n_jobs=1, return_estimator=True)
    fila = {"modelo": nombre, "n_feats": F.shape[1]}
    for m in METRICAS:
        fila[m] = r[f"test_{m}"].mean()
        fila[m + "_std"] = r[f"test_{m}"].std()
    if hasattr(r["estimator"][0], "best_params_"):
        fila["mejores_params"] = str([e.best_params_ for e in r["estimator"]])
    filas.append(fila)
    print(nombre, round(fila["accuracy"], 4), round(fila["accuracy_std"], 4), fila.get("mejores_params", ""), flush=True)
pd.DataFrame(filas).to_csv(RAIZ / "resultados" / "cv_hog16.csv", index=False)
