import math

import torch
import torch.nn.functional as F

MEDIA = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalizar(x, modo="global"):
    # x: float en [0,1], (N,3,H,W)
    if modo == "imagen":
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        s = x.std(dim=(1, 2, 3), keepdim=True).clamp_min(0.02)
        return (x - m) / s
    return (x - MEDIA) / STD


def a_tensor(X_uint8):
    return torch.from_numpy(X_uint8).permute(0, 3, 1, 2).float().div_(255.0)


def diedrico(x):
    # 8 orientaciones: el barco puede venir en cualquier rumbo
    out = torch.empty_like(x)
    k = torch.randint(0, 4, (x.shape[0],))
    flip = torch.rand(x.shape[0]) < 0.5
    for i in range(x.shape[0]):
        xi = torch.rot90(x[i], int(k[i]), dims=(1, 2))
        out[i] = xi.flip(2) if flip[i] else xi
    return out


def afin(x, rot=180, escala=(0.85, 1.15), desplaz=0.06):
    n = x.shape[0]
    ang = (torch.rand(n) * 2 - 1) * math.radians(rot)
    s = torch.empty(n).uniform_(*escala)
    tx = (torch.rand(n) * 2 - 1) * desplaz * 2
    ty = (torch.rand(n) * 2 - 1) * desplaz * 2
    cos, sin = torch.cos(ang) / s, torch.sin(ang) / s
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def color(x, fuerza=1.0):
    n = x.shape[0]
    u = lambda a, b: torch.empty(n, 1, 1, 1).uniform_(a, b)
    f = fuerza
    gris = x.mean(1, keepdim=True)
    x = gris + (x - gris) * u(1 - 0.35 * f, 1 + 0.35 * f)          # saturacion
    x = x * torch.empty(n, 3, 1, 1).uniform_(1 - 0.08 * f, 1 + 0.08 * f)  # balance de blancos
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - m) * u(1 - 0.3 * f, 1 + 0.3 * f) + m                  # contraste
    x = x * u(1 - 0.25 * f, 1 + 0.25 * f)                          # brillo
    x = x.clamp(0, 1) ** u(1 - 0.2 * f, 1 + 0.25 * f)               # gamma
    return x.clamp(0, 1)


def degradar(x, p_blur=0.3, p_ruido=0.3, p_resol=0.25):
    n, _, h, w = x.shape
    sel = torch.rand(n) < p_blur
    if sel.any():
        sigma = float(torch.empty(1).uniform_(0.4, 1.3))
        r = max(1, int(round(sigma * 2.5)))
        t = torch.arange(-r, r + 1).float()
        k = torch.exp(-t ** 2 / (2 * sigma ** 2)); k = k / k.sum()
        xs = x[sel]
        xs = F.conv2d(F.pad(xs, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(3, 1, 1, 1), groups=3)
        xs = F.conv2d(F.pad(xs, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(3, 1, 1, 1), groups=3)
        x = x.clone(); x[sel] = xs
    sel = torch.rand(n) < p_resol
    if sel.any():
        f = float(torch.empty(1).uniform_(0.45, 0.8))
        xs = F.interpolate(x[sel], scale_factor=f, mode="bilinear", align_corners=False, antialias=True)
        x = x.clone(); x[sel] = F.interpolate(xs, size=(h, w), mode="bilinear", align_corners=False)
    sel = torch.rand(n) < p_ruido
    if sel.any():
        x = x.clone()
        x[sel] = x[sel] + torch.randn_like(x[sel]) * float(torch.empty(1).uniform_(0.005, 0.03))
    return x.clamp(0, 1)


def aumentar(x, nivel="completo"):
    if nivel == "ninguno":
        return x
    x = diedrico(x)
    if nivel == "geometrico":
        return x
    x = afin(x, rot=45, escala=(0.85, 1.2), desplaz=0.05)
    x = color(x)
    return degradar(x)


def tta_8(x):
    vistas = []
    for k in range(4):
        r = torch.rot90(x, k, dims=(2, 3))
        vistas += [r, r.flip(3)]
    return vistas
