from pathlib import Path

import torch
import torch.nn as nn

PESOS_DIR = Path(__file__).resolve().parents[1] / "pesos_imagenet"

# pesos de timm publicados como releases en GitHub
PESOS_TIMM = {
    "resnet18d": "resnet18d_ra2-48a79e06.pth",
    "resnet26d": "resnet26d-69e92c46.pth",
    "efficientnet_b0": "efficientnet_b0_ra-3dd342df.pth",
    "mobilenetv3_large_100": "mobilenetv3_large_100_ra-f55367f5.pth",
    "convnext_atto": "convnext_atto_d2-01bb0f51.pth",
}


def bloque(cin, cout, drop=0.0):
    capas = [
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    ]
    if drop:
        capas.append(nn.Dropout2d(drop))
    return nn.Sequential(*capas)


class CNNBarcos(nn.Module):
    # 80 -> 40 -> 20 -> 10 -> 5, luego pooling global
    def __init__(self, ancho=32, dropout=0.3, n_bloques=4):
        super().__init__()
        chs = [ancho * 2 ** i for i in range(n_bloques)]
        capas, cin = [], 3
        for c in chs:
            capas.append(bloque(cin, c, drop=0.05))
            cin = c
        self.features = nn.Sequential(*capas)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(cin, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


class Timm(nn.Module):
    def __init__(self, nombre, preentrenado=True, dropout=0.2, lado=None):
        super().__init__()
        import timm
        self.net = timm.create_model(nombre, pretrained=False, num_classes=1, drop_rate=dropout)
        self.lado = lado
        if preentrenado:
            sd = torch.load(PESOS_DIR / PESOS_TIMM[nombre], map_location="cpu", weights_only=True)
            sd = {k: v for k, v in sd.items() if not k.startswith(("fc.", "classifier.", "head.fc."))}
            falt = self.net.load_state_dict(sd, strict=False)
            assert all(k.startswith(("fc.", "classifier.", "head.fc.")) for k in falt.missing_keys), falt

    def forward(self, x):
        if self.lado and x.shape[-1] != self.lado:
            x = nn.functional.interpolate(x, size=(self.lado, self.lado), mode="bilinear", align_corners=False)
        return self.net(x).squeeze(1)


def crear(nombre, **kw):
    if nombre == "cnn":
        return CNNBarcos(**kw)
    return Timm(nombre, **kw)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
