"""Descarga lo necesario para reentrenar: pesos ImageNet (timm) y el dataset ShipsNet.

ShipsNet se baja con la API de Kaggle (requiere ~/.kaggle/kaggle.json). Si no se tiene, basta con
descargar el zip a mano desde la pagina del dataset y descomprimirlo en data/.

uso: python descargar.py pesos | shipsnet
"""
import subprocess
import sys
import urllib.request
import zipfile

from datos import DATA_DIR, RAIZ
from redes import PESOS_DIR, PESOS_TIMM

BASE = "https://github.com/huggingface/pytorch-image-models/releases/download/"
RELEASE = {
    "resnet18d_ra2-48a79e06.pth": "v0.1-weights",
    "resnet26d-69e92c46.pth": "v0.1-weights",
    "efficientnet_b0_ra-3dd342df.pth": "v0.1-weights",
    "mobilenetv3_large_100_ra-f55367f5.pth": "v0.1-weights",
    "convnext_atto_d2-01bb0f51.pth": "v0.1-rsb-weights",
}


def pesos():
    PESOS_DIR.mkdir(exist_ok=True)
    for archivo in PESOS_TIMM.values():
        destino = PESOS_DIR / archivo
        if not destino.exists():
            print("bajando", archivo)
            urllib.request.urlretrieve(BASE + RELEASE[archivo] + "/" + archivo, destino)


def shipsnet():
    DATA_DIR.mkdir(exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", "rhammell/ships-in-satellite-imagery",
                    "-p", str(DATA_DIR)], check=True)
    with zipfile.ZipFile(DATA_DIR / "ships-in-satellite-imagery.zip") as z:
        z.extractall(DATA_DIR)
    # el zip trae shipsnet.json, la carpeta shipsnet/ y las escenas completas en scenes/
    from datos import cargar_shipsnet
    X, y = cargar_shipsnet()
    print("ShipsNet:", X.shape, "barcos:", int(y.sum()))


if __name__ == "__main__":
    for arg in sys.argv[1:] or ["pesos", "shipsnet"]:
        {"pesos": pesos, "shipsnet": shipsnet}[arg]()
    print("listo, datos en", DATA_DIR, "y pesos en", RAIZ / "pesos_imagenet")
