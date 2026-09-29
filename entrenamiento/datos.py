import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("SHIPS_DATA", RAIZ / "data"))
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SI = {"1", "ship", "ships", "barco", "barcos", "positivo", "positivos"}
NO = {"0", "no_ship", "no-ship", "no ship", "noship", "no_ships", "not_ship", "no_barco", "no-barco",
      "no barco", "nobarco", "no_barcos", "sin_barco", "sin barco", "negativo", "negativos"}


def leer_imagen(ruta, lado=80):
    img = Image.open(ruta).convert("RGB")
    if img.size != (lado, lado):
        img = img.resize((lado, lado), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def etiqueta_desde_nombre(ruta):
    # formato de ShipsNet: {label}__{scene}__{lon}_{lat}.png, o carpetas barco/no_barco
    nombre = Path(ruta).name
    m = re.match(r"^([01])__", nombre)
    if m:
        return int(m.group(1))
    padre = Path(ruta).parent.name.lower()
    if padre in SI:
        return 1
    if padre in NO:
        return 0
    return None


def cargar_carpeta(carpeta):
    rutas = sorted(p for p in Path(carpeta).rglob("*") if p.suffix.lower() in EXTS)
    X = np.stack([leer_imagen(p) for p in rutas]) if rutas else np.zeros((0, 80, 80, 3), np.uint8)
    y = np.array([etiqueta_desde_nombre(p) if etiqueta_desde_nombre(p) is not None else -1 for p in rutas])
    return X, y, rutas


def cargar_json(ruta):
    with open(ruta) as f:
        d = json.load(f)
    X = np.array(d["data"], dtype=np.uint8).reshape(-1, 3, 80, 80).transpose(0, 2, 3, 1)
    y = np.array(d["labels"], dtype=np.int64)
    return X, y


def cargar_shipsnet():
    cache = DATA_DIR / "shipsnet.npz"
    if cache.exists():
        d = np.load(cache)
        return d["X"], d["y"]
    if (DATA_DIR / "shipsnet.json").exists():
        X, y = cargar_json(DATA_DIR / "shipsnet.json")
    elif (DATA_DIR / "shipsnet").exists():
        X, y, _ = cargar_carpeta(DATA_DIR / "shipsnet")
    else:
        raise FileNotFoundError(
            f"No encuentro ShipsNet en {DATA_DIR}. Descargar de "
            "https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery"
        )
    np.savez_compressed(cache, X=X, y=y)
    return X, y


def ruta_npz(nombre):
    # primero data/, luego los recortes que vienen con el repositorio
    for carpeta in (DATA_DIR, RAIZ / "datos_externos"):
        if (carpeta / nombre).exists():
            return carpeta / nombre
    return None


def cargar_npz(nombre):
    d = np.load(ruta_npz(nombre))
    return d["X"], d["y"]
