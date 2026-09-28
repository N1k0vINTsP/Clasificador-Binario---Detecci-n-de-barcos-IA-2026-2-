import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from aumentos import normalizar


class Envoltorio(nn.Module):
    # recibe la imagen en [0,1] y devuelve p(barco); si hay varias redes promedia
    def __init__(self, modelos, normas):
        super().__init__()
        self.modelos = nn.ModuleList(modelos)
        self.normas = list(normas)

    def forward(self, x):
        ps = [torch.sigmoid(m(normalizar(x, n))) for m, n in zip(self.modelos, self.normas)]
        return torch.stack(ps, 0).mean(0)


def exportar(modelos, normas, ruta_onnx):
    env = Envoltorio([m.float().eval().to(memory_format=torch.contiguous_format) for m in modelos], normas).eval()
    x = torch.rand(2, 3, 80, 80)
    torch.onnx.export(env, x, str(ruta_onnx), input_names=["imagen"], output_names=["p_barco"],
                      dynamic_axes={"imagen": {0: "n"}, "p_barco": {0: "n"}}, opset_version=17, dynamo=False)
    # comprobar que ONNX da lo mismo que PyTorch
    s = ort.InferenceSession(str(ruta_onnx), providers=["CPUExecutionProvider"])
    x = torch.rand(16, 3, 80, 80)
    with torch.no_grad():
        ref = env(x).numpy()
    out = s.run(None, {"imagen": x.numpy()})[0]
    dif = float(np.abs(ref - out).max())
    xb = np.random.rand(64, 3, 80, 80).astype(np.float32)
    s.run(None, {"imagen": xb})
    t = time.perf_counter()
    for _ in range(5):
        s.run(None, {"imagen": xb})
    ms = (time.perf_counter() - t) / (5 * 64) * 1000
    return {"dif_max_torch_onnx": dif, "ms_por_imagen_cpu": ms,
            "tamano_mb": Path(ruta_onnx).stat().st_size / 1e6}


def guardar_info(carpeta, archivo, extra):
    info = {"archivo": archivo, "entrada": "RGB 80x80, valores 0-1, formato NCHW", "umbral": 0.5}
    info.update(extra)
    Path(carpeta, "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
