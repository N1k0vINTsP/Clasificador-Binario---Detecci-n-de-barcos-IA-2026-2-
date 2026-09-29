"""Costo de inferencia de cada arquitectura en ONNX Runtime sobre CPU (1 hilo, como en un
computador embarcado modesto). Mide una imagen sola y las 8 vistas del TTA."""
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

from datos import RAIZ
from exportar import Envoltorio
from redes import crear, n_params

import torch


def medir(ruta, lote, repeticiones=200):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    s = ort.InferenceSession(str(ruta), opts, providers=["CPUExecutionProvider"])
    x = np.random.rand(lote, 3, 80, 80).astype(np.float32)
    for _ in range(10):
        s.run(None, {"imagen": x})
    t = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        s.run(None, {"imagen": x})
        t.append(time.perf_counter() - t0)
    return 1000 * float(np.median(t))


def main():
    filas = []
    modelos = [("CNN propia", "cnn", dict(ancho=32)), ("ResNet18-D", "resnet18d", dict(preentrenado=False)),
               ("MobileNetV3-L", "mobilenetv3_large_100", dict(preentrenado=False)),
               ("ConvNeXt-Atto", "convnext_atto", dict(preentrenado=False)),
               ("EfficientNet-B0", "efficientnet_b0", dict(preentrenado=False))]
    with tempfile.TemporaryDirectory() as tmp:
        for nombre, red, kw in modelos:
            m = crear(red, **kw).eval()
            env = Envoltorio([m], ["global"]).eval()
            ruta = Path(tmp) / f"{red}.onnx"
            torch.onnx.export(env, torch.rand(1, 3, 80, 80), str(ruta), input_names=["imagen"],
                              output_names=["p_barco"], dynamic_axes={"imagen": {0: "n"}, "p_barco": {0: "n"}},
                              opset_version=17, dynamo=False)
            fila = {"modelo": nombre, "parametros_M": n_params(m) / 1e6, "onnx_MB": ruta.stat().st_size / 1e6,
                    "ms_1_imagen": medir(ruta, 1), "ms_tta_8": medir(ruta, 8)}
            filas.append(fila)
            print(fila, flush=True)
    pd.DataFrame(filas).to_csv(RAIZ / "resultados" / "latencia.csv", index=False)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
