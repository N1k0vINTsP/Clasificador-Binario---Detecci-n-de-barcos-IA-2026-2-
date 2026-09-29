import copy
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from aumentos import a_tensor, aumentar, normalizar, tta_8

torch.set_num_threads(int(os.environ.get("HILOS", "4")))


def fijar_semilla(s):
    torch.manual_seed(s)
    np.random.seed(s)


def _ctx(bf16):
    # el Xeon del servidor tiene AMX, en bf16 el entrenamiento va varias veces mas rapido
    return torch.autocast("cpu", dtype=torch.bfloat16) if bf16 else nullcontext()


def entrenar(modelo, Xtr, ytr, epocas=40, lr=2e-3, wd=5e-4, bs=64, aug="completo", norm="global",
             suavizado=0.05, balancear=True, semilla=0, Xva=None, yva=None, log=True, ema=0.0, bf16=True):
    fijar_semilla(semilla)
    modelo = modelo.to(memory_format=torch.channels_last)
    xt = a_tensor(Xtr)
    yt = torch.from_numpy(ytr).float()
    n = len(yt)
    pw = ((n - yt.sum()) / yt.sum()).clone() if balancear else None
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=wd)
    pasos = epocas * ((n + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=pasos, pct_start=0.15)
    modelo_ema = copy.deepcopy(modelo).eval() if ema else None
    hist = []
    for ep in range(epocas):
        modelo.train()
        t0 = time.time()
        perm = torch.randperm(n)
        perdida = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            xb = normalizar(aumentar(xt[idx], aug), norm).contiguous(memory_format=torch.channels_last)
            yb = yt[idx]
            obj = yb * (1 - suavizado) + 0.5 * suavizado
            with _ctx(bf16):
                logit = modelo(xb)
            loss = F.binary_cross_entropy_with_logits(logit.float(), obj, pos_weight=pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if modelo_ema is not None:
                with torch.no_grad():
                    for pe, pm in zip(modelo_ema.parameters(), modelo.parameters()):
                        pe.mul_(ema).add_(pm.detach(), alpha=1 - ema)
                    for be, bm in zip(modelo_ema.buffers(), modelo.buffers()):
                        be.copy_(bm)
            perdida += loss.item() * len(idx)
        fila = {"epoca": ep + 1, "loss": perdida / n, "seg": time.time() - t0}
        evaluar_ahora = (ep + 1) % 5 == 0 or ep + 1 == epocas
        if Xva is not None and evaluar_ahora:
            p = predecir(modelo_ema or modelo, Xva, norm, bf16=bf16)
            fila["val_acc"] = float(((p >= 0.5) == yva).mean())
        hist.append(fila)
        if log:
            print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in fila.items()), flush=True)
    return (modelo_ema or modelo), hist


@torch.no_grad()
def predecir(modelo, X, norm="global", tta=False, bs=256, bf16=False):
    modelo.eval()
    probs = []
    for i in range(0, len(X), bs):
        xb = a_tensor(X[i:i + bs])
        vistas = tta_8(xb) if tta else [xb]
        ps = []
        for v in vistas:
            with _ctx(bf16):
                ps.append(torch.sigmoid(modelo(normalizar(v, norm).contiguous(memory_format=torch.channels_last)).float()))
        probs.append(torch.stack(ps).mean(0))
    return torch.cat(probs).numpy()
