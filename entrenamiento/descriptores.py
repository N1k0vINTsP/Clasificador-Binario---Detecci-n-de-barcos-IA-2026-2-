import numpy as np
from joblib import Parallel, delayed
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog, local_binary_pattern
from skimage.transform import resize


def pixeles(img, lado=40):
    return resize(img, (lado, lado), anti_aliasing=True).ravel()


def hist_color(img, bins=16):
    hsv = rgb2hsv(img)
    partes = [np.histogram(hsv[..., c], bins=bins, range=(0, 1), density=True)[0] for c in range(3)]
    # momentos por canal RGB, ayudan con el cambio de iluminacion entre escenas
    rgb = img.reshape(-1, 3) / 255.0
    momentos = np.concatenate([rgb.mean(0), rgb.std(0)])
    return np.concatenate(partes + [momentos])


def hog_desc(img, celda=8, orient=9, bloque=2):
    g = rgb2gray(img)
    return hog(g, orientations=orient, pixels_per_cell=(celda, celda),
               cells_per_block=(bloque, bloque), block_norm="L2-Hys", feature_vector=True)


def lbp_desc(img, P=8, R=1):
    g = (rgb2gray(img) * 255).astype(np.uint8)
    lbp = local_binary_pattern(g, P, R, method="uniform")
    n = P + 2
    # histograma global + 2x2 regiones para conservar algo de la posicion
    hs = [np.histogram(lbp, bins=n, range=(0, n), density=True)[0]]
    for i in (0, 40):
        for j in (0, 40):
            hs.append(np.histogram(lbp[i:i + 40, j:j + 40], bins=n, range=(0, n), density=True)[0])
    return np.concatenate(hs)


EXTRACTORES = {
    "pixeles": pixeles,
    "color": hist_color,
    "hog": hog_desc,
    "lbp": lbp_desc,
}


def extraer(X, nombres, n_jobs=4, **kw):
    def una(img):
        return np.concatenate([EXTRACTORES[n](img, **kw.get(n, {})) for n in nombres])
    feats = Parallel(n_jobs=n_jobs, batch_size=64)(delayed(una)(img) for img in X)
    return np.asarray(feats, dtype=np.float32)
