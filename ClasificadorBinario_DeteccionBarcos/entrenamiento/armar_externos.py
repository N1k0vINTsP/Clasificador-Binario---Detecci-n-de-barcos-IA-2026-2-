"""Reconstruye los conjuntos adicionales a partir de las coordenadas guardadas en datos_externos/.

- ext_planet.npz: recortes nuevos de las escenas completas de Planet (carpeta scenes/ del Kaggle).
  Solo para prueba.
- ext_rotterdam.npz: recortes del puerto de Rotterdam (SpaceNet 6, WorldView-2 a 0.5 m), llevados a
  80x80. Solo para prueba: nunca entra al entrenamiento.
- ext_maxar.npz: recortes de 9 escenas de 7 puertos (Valencia, Colombo, Tampa, Kingston en dos fechas,
  Iskenderun, Durban, Ravenna) del programa Maxar Open Data, a 1.2 m. Se usa para entrenar.

Las etiquetas se revisaron una por una a ojo. Los candidatos a barco de Rotterdam salieron de un
detector YOLO publico entrenado con imagenes de Google Earth; aqui solo se guardan las coordenadas.

uso: python armar_externos.py [planet] [rotterdam] [maxar]
"""
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from datos import DATA_DIR, RAIZ

EXT = RAIZ / "datos_externos"
S3 = "https://spacenet-dataset.s3.amazonaws.com/spacenet/SN6_buildings/train/AOI_11_Rotterdam/PS-RGB/"


def planet():
    J = json.loads((EXT / "planet_escenas.json").read_text())
    escenas = {}
    X, y, grupo = [], [], []
    for c in J["chips"]:
        if c["escena"] not in escenas:
            # en el zip de Kaggle las escenas quedan en scenes/scenes/
            ruta = next((DATA_DIR / "scenes").rglob(f"{c['escena']}.png"))
            escenas[c["escena"]] = np.asarray(Image.open(ruta).convert("RGB"))
        a = escenas[c["escena"]]
        X.append(a[c["y0"]:c["y0"] + 80, c["x0"]:c["x0"] + 80])
        y.append(c["etiqueta"])
        grupo.append(c["grupo"])
    np.savez_compressed(DATA_DIR / "ext_planet.npz", X=np.stack(X), y=np.array(y), sub=np.array(grupo))
    print("ext_planet:", len(y), "imagenes,", sum(y), "barcos")


def mosaico_rotterdam(J):
    import tifffile
    carpeta = DATA_DIR / "sn6_psrgb"
    carpeta.mkdir(exist_ok=True)
    x0, y1 = J["mosaico"]["x0_utm"], J["mosaico"]["y1_utm"]
    tiles = []
    for n in J["tiles"]:
        ruta = carpeta / n
        if not ruta.exists():
            urllib.request.urlretrieve(S3 + n, ruta)
        with tifffile.TiffFile(ruta) as t:
            tp = t.pages[0].tags["ModelTiepointTag"].value
        tiles.append((ruta, int(round(tp[3] - x0)), int(round(y1 - tp[4]))))
    W = max(c for _, c, _ in tiles) + 450
    H = max(r for _, _, r in tiles) + 450
    mos = np.zeros((H, W, 3), np.uint8)
    for ruta, c, r in tiles:
        a = tifffile.imread(ruta).transpose(1, 2, 0)
        a = np.asarray(Image.fromarray(a).resize((450, 450), Image.BOX))
        sub = mos[r:r + 450, c:c + 450]
        a = a[:sub.shape[0], :sub.shape[1]]
        v = a.sum(2) > 0
        sub[v] = a[v]
    return mos


def rotterdam():
    J = json.loads((EXT / "rotterdam_sn6.json").read_text())
    mos = mosaico_rotterdam(J)
    X, y, grupo = [], [], []
    for c in J["chips"]:
        S = int(round(c["lado_m"]))
        x0 = int(round(c["cx"] - S / 2))
        y0 = int(round(c["cy"] - S / 2))
        rec = Image.fromarray(mos[y0:y0 + S, x0:x0 + S]).resize((80, 80), Image.LANCZOS)
        X.append(np.asarray(rec))
        y.append(c["etiqueta"])
        grupo.append(c["grupo"])
    np.savez_compressed(DATA_DIR / "ext_rotterdam.npz", X=np.stack(X), y=np.array(y), sub=np.array(grupo))
    print("ext_rotterdam:", len(y), "imagenes,", sum(y), "barcos")


def ventana_maxar(url, lon0, lat0, lon1, lat1, res=1.2):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform
    from rasterio.windows import from_bounds
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(url) as src:
            xs, ys = transform("EPSG:4326", src.crs, [lon0, lon1], [lat0, lat1])
            win = from_bounds(min(xs), min(ys), max(xs), max(ys), src.transform).round_offsets().round_lengths()
            h = int(round(win.height * src.res[1] / res))
            w = int(round(win.width * src.res[0] / res))
            a = src.read(window=win, out_shape=(3, h, w), resampling=Resampling.average, boundless=True, fill_value=0)
    return a.transpose(1, 2, 0)


def maxar():
    J = json.loads((EXT / "maxar_puertos.json").read_text())
    X, y, grupo, puerto = [], [], [], []
    for nombre, esc in J["escenas"].items():
        img = ventana_maxar(esc["url"], *esc["lon_lat"], res=J["resolucion_m"])
        for c in (c for c in J["chips"] if c["escena"] == nombre):
            S = int(round(c["lado"]))
            x0 = int(round(c["cx"] - S / 2))
            y0 = int(round(c["cy"] - S / 2))
            X.append(np.asarray(Image.fromarray(img[y0:y0 + S, x0:x0 + S]).resize((80, 80), Image.LANCZOS)))
            y.append(c["etiqueta"])
            grupo.append(f"{nombre}_{c['grupo']}")
            puerto.append(nombre)
        print(nombre, "ok", flush=True)
    np.savez_compressed(DATA_DIR / "ext_maxar.npz", X=np.stack(X), y=np.array(y), sub=np.array(puerto),
                        grupo=np.array(grupo))
    print("ext_maxar:", len(y), "imagenes,", sum(y), "con barco")


if __name__ == "__main__":
    for arg in sys.argv[1:] or ["planet", "rotterdam", "maxar"]:
        {"planet": planet, "rotterdam": rotterdam, "maxar": maxar}[arg]()
