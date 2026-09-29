import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MODELO_DIR = Path(__file__).resolve().parent / "modelo"
LADO = 80


def cargar_imagen(fuente):
    img = Image.open(fuente)
    original = img.size
    if img.mode in ("I;16", "I;16B", "I", "F"):
        # tif de 16 bits o flotante: se estira al rango 0-255
        a = np.asarray(img, dtype=np.float32)
        lo, hi = np.percentile(a, [0.5, 99.5])
        a = np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(a)
    img = img.convert("RGB")
    if img.size != (LADO, LADO):
        img = img.resize((LADO, LADO), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8), original


def vistas_tta(x):
    # x: (N,3,H,W). Rotaciones de 90° y reflejos: el barco no tiene una orientación preferida
    out = []
    for k in range(4):
        r = np.rot90(x, k, axes=(2, 3))
        out += [r, r[..., ::-1]]
    return out


class Clasificador:
    def __init__(self, carpeta=MODELO_DIR):
        carpeta = Path(carpeta)
        self.info = json.loads((carpeta / "info.json").read_text(encoding="utf-8"))
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.sesion = ort.InferenceSession(str(carpeta / self.info["archivo"]), opts,
                                           providers=["CPUExecutionProvider"])
        self.entrada = self.sesion.get_inputs()[0].name
        self.umbral = float(self.info.get("umbral", 0.5))

    def probabilidades(self, imgs, tta=True, lote=64):
        # imgs: uint8 (N, 80, 80, 3) -> p(barco) por imagen
        imgs = np.asarray(imgs)
        salida = []
        for i in range(0, len(imgs), lote):
            x = imgs[i:i + lote].transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            vistas = vistas_tta(x) if tta else [x]
            ps = [self.sesion.run(None, {self.entrada: np.ascontiguousarray(v)})[0].reshape(-1) for v in vistas]
            salida.append(np.mean(ps, axis=0))
        return np.concatenate(salida) if salida else np.zeros(0)

    def predecir(self, imgs, tta=True, umbral=None):
        p = self.probabilidades(imgs, tta)
        return (p >= (self.umbral if umbral is None else umbral)).astype(int), p
