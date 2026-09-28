import datetime as dt
import io
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps

from clasificador import Clasificador, cargar_imagen

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SI = {"1", "ship", "ships", "barco", "barcos", "positivo", "positivos"}
NO = {"0", "no_ship", "no-ship", "no ship", "noship", "no_ships", "not_ship", "no_barco", "no-barco",
      "no barco", "nobarco", "no_barcos", "sin_barco", "sin barco", "negativo", "negativos"}
RAIZ = Path(__file__).resolve().parent
VERDE, ROJO, GRIS = "#2e7d32", "#c62828", "#9e9e9e"

st.set_page_config(page_title="Barcos UAV · evaluación en vivo", layout="wide")


@st.cache_resource
def cargar_modelo():
    return Clasificador()


def etiqueta_por_nombre(ruta):
    nombre = Path(ruta).name.lower()
    m = re.match(r"^([01])__", nombre)
    if m:
        return int(m.group(1))
    padre = Path(ruta).parent.name.lower()
    if padre in SI:
        return 1
    if padre in NO:
        return 0
    return -1


def elegir_carpeta():
    # el dialogo corre en otro proceso para no pelear con el hilo de streamlit (en macOS se cae)
    codigo = ("import tkinter as tk\nfrom tkinter import filedialog\nr = tk.Tk(); r.withdraw()\n"
              "r.attributes('-topmost', True)\nprint(filedialog.askdirectory(title='Carpeta de prueba'))")
    try:
        out = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, timeout=300)
        return out.stdout.strip()
    except Exception:
        return ""


def nuevo_lote(nombres, imgs, tamanos, etq_nombre, origen):
    st.session_state.update(
        nombres=nombres, imgs=np.stack(imgs), tamanos=tamanos, etq_nombre=np.array(etq_nombre),
        etq=np.full(len(nombres), -1), orden=[], prob=None, i=0, origen=origen,
        lote=st.session_state.get("lote", 0) + 1, inicio=dt.datetime.now())


def leer(fuentes):
    nombres, imgs, tamanos, etq = [], [], [], []
    for nombre, fuente in fuentes:
        try:
            a, tam = cargar_imagen(fuente)
        except Exception:
            continue
        nombres.append(nombre)
        imgs.append(a)
        tamanos.append(tam)
        etq.append(etiqueta_por_nombre(nombre))
    return nombres, imgs, tamanos, etq


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - m), min(1.0, c + m)


def metricas(y, yp):
    tp = int(((y == 1) & (yp == 1)).sum())
    tn = int(((y == 0) & (yp == 0)).sum())
    fp = int(((y == 0) & (yp == 1)).sum())
    fn = int(((y == 1) & (yp == 0)).sum())
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp else (0.0 if n and (fp or fn) else float("nan"))
    return dict(n=n, tp=tp, tn=tn, fp=fp, fn=fn, acc=acc, prec=prec, rec=rec, f1=f1)


def pct(x, dec=1):
    return "—" if x != x else f"{100 * x:.{dec}f} %"


def ampliar(a, lado=280, borde=None):
    img = Image.fromarray(a).resize((lado, lado), Image.NEAREST)
    return ImageOps.expand(img, border=6, fill=borde) if borde else img


def atajos_teclado():
    # b = barco, n = no barco, flechas = navegar. Si el navegador lo bloquea, quedan los botones.
    components.html("""
    <script>
    const doc = window.parent.document;
    if (!doc.__atajosBarcos) {
      doc.__atajosBarcos = true;
      doc.addEventListener('keydown', (e) => {
        if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
        const mapa = {b: 'Barco', n: 'No barco', ArrowLeft: '◀ Anterior', ArrowRight: 'Siguiente ▶'};
        const texto = mapa[e.key];
        if (!texto) return;
        const btn = [...doc.querySelectorAll('button')].find(x => x.innerText.trim() === texto);
        if (btn && !btn.disabled) { btn.click(); e.preventDefault(); }
      });
    }
    </script>""", height=0)


modelo = cargar_modelo()
info = modelo.info
cv = info.get("cv", {})

# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.subheader("Imágenes de prueba", anchor=False)
    if "carpeta" not in st.session_state:
        st.session_state.carpeta = str(RAIZ / "test")
    if st.button("Elegir carpeta…", use_container_width=True):
        elegida = elegir_carpeta()
        if elegida:
            st.session_state.carpeta = elegida
    carpeta = st.text_input("Carpeta", key="carpeta")
    if st.button("Cargar carpeta", type="primary", use_container_width=True):
        base = Path(carpeta)
        if not base.is_dir():
            st.error("No existe esa carpeta.")
        else:
            rutas = sorted(p for p in base.rglob("*") if p.suffix.lower() in EXTS)
            datos = leer([(str(p.relative_to(base)), p) for p in rutas])
            if datos[0]:
                nuevo_lote(*datos, origen=str(base))
            else:
                st.warning("No encontré imágenes (.png, .jpg, .tif, .bmp) en esa carpeta.")
    subidas = st.file_uploader("…o arrastrar las imágenes aquí", accept_multiple_files=True,
                               type=[e[1:] for e in EXTS])
    if subidas and st.button("Usar imágenes subidas", use_container_width=True):
        nuevo_lote(*leer([(f.name, io.BytesIO(f.getvalue())) for f in subidas]), origen="archivos subidos")

    st.divider()
    umbral = st.slider("Umbral p(barco)", 0.05, 0.95, float(modelo.umbral), 0.05)
    tta = st.toggle("TTA (8 orientaciones)", value=True,
                    help="Promedia la salida de la imagen rotada 0/90/180/270° y reflejada.")
    a_ciegas = st.toggle("Ocultar la predicción hasta etiquetar", value=True)

    st.divider()
    if cv:
        st.caption("Validación cruzada del modelo (5 folds)")
        st.markdown(f"Accuracy **{100 * cv['accuracy']:.2f} %** ± {100 * cv['accuracy_std']:.2f}  \n"
                    f"Precisión {100 * cv['precision']:.2f} % · Recall {100 * cv['recall']:.2f} %")
    st.caption(info.get("descripcion", info["archivo"]))

st.title("Detección de barcos en imágenes aéreas", anchor=False)
st.caption("Clasificador barco / no barco (RGB 80×80) del sistema de percepción del UAV · evaluación en vivo")

if "imgs" not in st.session_state:
    st.info("Elija la carpeta con las imágenes de prueba en la barra lateral y pulse **Cargar carpeta**.")
    st.stop()

imgs, nombres = st.session_state.imgs, st.session_state.nombres
N = len(nombres)

clave = (st.session_state.lote, tta)
if st.session_state.get("clave_inf") != clave:
    with st.spinner(f"Clasificando {N} imágenes…"):
        t0 = dt.datetime.now()
        st.session_state.prob = modelo.probabilidades(imgs, tta=tta)
        st.session_state.ms_img = (dt.datetime.now() - t0).total_seconds() * 1000 / N
    st.session_state.clave_inf = clave
prob = st.session_state.prob
pred = (prob >= umbral).astype(int)
etq = st.session_state.etq

if (st.session_state.etq_nombre >= 0).any() and (etq < 0).all():
    c1, c2 = st.columns([4, 1])
    c1.info(f"{int((st.session_state.etq_nombre >= 0).sum())} de {N} imágenes traen la etiqueta "
            "en el nombre del archivo o en la carpeta (barco/no_barco, 1/0).")
    if c2.button("Usar esas etiquetas", use_container_width=True):
        st.session_state.etq = st.session_state.etq_nombre.copy()
        st.session_state.orden = [int(k) for k in np.where(st.session_state.etq >= 0)[0]]
        st.rerun()

# ---------------------------------------------------------------- métricas en vivo
hechas = etq >= 0
m = metricas(etq[hechas], pred[hechas])
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Etiquetadas", f"{m['n']} / {N}")
delta = f"{100 * (m['acc'] - cv['accuracy']):+.1f} pts vs CV" if cv and m["n"] else None
k2.metric("Accuracy en vivo", pct(m["acc"]), delta=delta)
k3.metric("Precisión", pct(m["prec"]))
k4.metric("Recall", pct(m["rec"]))
k5.metric("F1", pct(m["f1"]))
if m["n"]:
    lo, hi = wilson(m["tp"] + m["tn"], m["n"])
    st.caption(f"IC 95 % del accuracy (Wilson): {100 * lo:.1f} – {100 * hi:.1f} % · "
               f"inferencia ≈ {st.session_state.ms_img:.1f} ms por imagen en este equipo")

izq, der = st.columns([1, 1.1], gap="large")

with izq:
    i = st.session_state.i
    st.subheader(f"Imagen {i + 1} de {N}", anchor=False)
    borde = None if etq[i] < 0 else (VERDE if etq[i] == pred[i] else ROJO)
    st.image(ampliar(imgs[i], borde=borde), caption=nombres[i])
    if st.session_state.tamanos[i] != (80, 80):
        w, h = st.session_state.tamanos[i]
        st.caption(f"Original de {w}×{h} px, llevada a 80×80.")

    def poner(valor):
        k = st.session_state.i
        if st.session_state.etq[k] < 0:
            st.session_state.orden.append(k)
        st.session_state.etq[k] = valor
        pend = np.where(st.session_state.etq < 0)[0]
        if len(pend):
            sig = pend[pend > k]
            st.session_state.i = int(sig[0] if len(sig) else pend[0])

    def mover(paso):
        st.session_state.i = int(np.clip(st.session_state.i + paso, 0, N - 1))

    def borrar():
        k = st.session_state.i
        st.session_state.etq[k] = -1
        st.session_state.orden = [j for j in st.session_state.orden if j != k]

    b1, b2 = st.columns(2)
    b1.button("Barco", on_click=poner, args=(1,), use_container_width=True, type="primary")
    b2.button("No barco", on_click=poner, args=(0,), use_container_width=True)
    n1, n2, n3 = st.columns(3)
    n1.button("◀ Anterior", on_click=mover, args=(-1,), use_container_width=True, disabled=i == 0)
    n2.button("Siguiente ▶", on_click=mover, args=(1,), use_container_width=True, disabled=i == N - 1)
    n3.button("Borrar etiqueta", on_click=borrar, use_container_width=True, disabled=etq[i] < 0)
    st.session_state.ir_a = i + 1
    c1, c2 = st.columns([1, 2])
    c1.number_input("Ir a la imagen", min_value=1, max_value=N, step=1, key="ir_a",
                    on_change=lambda: st.session_state.update(i=int(st.session_state.ir_a) - 1))
    c2.caption("Teclado: **b** barco · **n** no barco · ← → para moverse")

    if a_ciegas and etq[i] < 0:
        st.markdown("Modelo: *(se muestra al etiquetar)*")
    else:
        texto = "BARCO" if pred[i] else "NO BARCO"
        extra = "" if etq[i] < 0 else ("  ✓ acierto" if etq[i] == pred[i] else "  ✗ error")
        st.markdown(f"Modelo: **{texto}** · p(barco) = {prob[i]:.3f}{extra}")

with der:
    st.subheader("Matriz de confusión", anchor=False)
    cm = pd.DataFrame([["barco", "barco", m["tp"]], ["barco", "no barco", m["fn"]],
                       ["no barco", "barco", m["fp"]], ["no barco", "no barco", m["tn"]]],
                      columns=["real", "predicho", "n"])
    base = alt.Chart(cm).encode(
        x=alt.X("predicho:N", sort=["barco", "no barco"], title="predicho",
                axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y("real:N", sort=["barco", "no barco"], title="real"))
    celdas = base.mark_rect(stroke="white", strokeWidth=2).encode(
        color=alt.Color("n:Q", scale=alt.Scale(scheme="blues", domainMin=0), legend=None),
        tooltip=["real", "predicho", "n"])
    numeros = base.mark_text(fontSize=24, fontWeight="bold").encode(
        text="n:Q", color=alt.condition(f"datum.n > {max(1, m['n']) * 0.45}", alt.value("white"), alt.value("#1d2733")))
    st.altair_chart((celdas + numeros).properties(height=190), use_container_width=True)

    if len(st.session_state.orden) >= 2:
        orden = np.array(st.session_state.orden)
        ok = (etq[orden] == pred[orden]).astype(float)
        n_ = np.arange(1, len(ok) + 1)
        curva = pd.DataFrame({"etiquetadas": n_, "accuracy": np.cumsum(ok) / n_})
        piso = min(0.9, float(curva.accuracy.min()) - 0.02)
        capas = [alt.Chart(curva).mark_line(strokeWidth=2, color="#1f4e79").encode(
            x=alt.X("etiquetadas:Q", title="imágenes etiquetadas", axis=alt.Axis(tickMinStep=1, format="d")),
            y=alt.Y("accuracy:Q", title="accuracy acumulado", scale=alt.Scale(domain=[piso, 1.0]),
                    axis=alt.Axis(format="%")),
            tooltip=["etiquetadas", alt.Tooltip("accuracy:Q", format=".1%")])]
        capas.append(alt.Chart(pd.DataFrame({"y": [0.98]})).mark_rule(color=ROJO, opacity=0.7).encode(y="y:Q"))
        if cv:
            capas.append(alt.Chart(pd.DataFrame({"y": [cv["accuracy"]]})).mark_rule(color=GRIS).encode(y="y:Q"))
        st.altair_chart(alt.layer(*capas).properties(height=190), use_container_width=True)
        st.caption("Rojo: meta del 98 % · gris: accuracy de la validación cruzada")

    if cv:
        st.markdown("**Validación cruzada vs. prueba en vivo**")
        comp = pd.DataFrame({
            "métrica": ["Accuracy", "Precisión", "Recall"],
            "validación cruzada": [pct(cv["accuracy"], 2), pct(cv["precision"], 2), pct(cv["recall"], 2)],
            "en vivo": [pct(m["acc"]), pct(m["prec"]), pct(m["rec"])],
        })
        st.dataframe(comp, hide_index=True, use_container_width=True)
        if m["n"] >= 10:
            lo, hi = wilson(m["tp"] + m["tn"], m["n"])
            if lo <= cv["accuracy"] <= hi:
                st.caption(f"El accuracy de la validación cruzada ({100 * cv['accuracy']:.1f} %) cae dentro del "
                           f"IC 95 % de la prueba en vivo ({100 * lo:.1f}–{100 * hi:.1f} %): no hay evidencia de que "
                           "el desempeño cambie con estas imágenes.")
            elif cv["accuracy"] > hi:
                st.caption(f"El accuracy de la validación cruzada ({100 * cv['accuracy']:.1f} %) queda por encima del "
                           f"IC 95 % en vivo ({100 * lo:.1f}–{100 * hi:.1f} %): estas imágenes difieren de las de "
                           "entrenamiento (sensor, escala o criterio de etiquetado). Revisar la pestaña de errores.")
            else:
                st.caption(f"En vivo el modelo rinde por encima de lo que predice la validación cruzada "
                           f"(IC 95 % {100 * lo:.1f}–{100 * hi:.1f} %).")

# ---------------------------------------------------------------- resultados
tabla = pd.DataFrame({
    "archivo": nombres,
    "p_barco": np.round(prob, 4),
    "prediccion": np.where(pred == 1, "barco", "no barco"),
    "etiqueta": np.select([etq == 1, etq == 0], ["barco", "no barco"], ""),
})
tabla["correcto"] = np.where(etq < 0, "", np.where(etq == pred, "sí", "no"))

st.divider()
tab1, tab2 = st.tabs(["Galería", "Tabla"])
with tab1:
    filtro = st.radio("Mostrar", ["todas", "sin etiquetar", "errores"], horizontal=True, label_visibility="collapsed")
    idx = np.arange(N)
    if filtro == "sin etiquetar":
        idx = idx[etq < 0]
    elif filtro == "errores":
        idx = idx[(etq >= 0) & (etq != pred)]
    if len(idx) > 200:
        st.caption(f"Se muestran las primeras 200 de {len(idx)}.")
    cols = st.columns(12)
    for j, k in enumerate(idx[:200]):
        color = GRIS if etq[k] < 0 else (VERDE if etq[k] == pred[k] else ROJO)
        texto = f"{k + 1}" if a_ciegas and etq[k] < 0 else f"{k + 1} · {'B' if pred[k] else 'N'} {prob[k]:.2f}"
        cols[j % 12].image(ampliar(imgs[k], 88, borde=color), caption=texto)
    st.caption("Borde verde: acierto · rojo: error · gris: sin etiquetar. B = barco, N = no barco.")
with tab2:
    vista = tabla.copy()
    if a_ciegas:
        vista.loc[etq < 0, ["p_barco", "prediccion"]] = [np.nan, ""]
    st.dataframe(vista, use_container_width=True, hide_index=True)

c1, c2 = st.columns([1, 3])
c1.download_button("Descargar CSV", tabla.to_csv(index=False).encode("utf-8"),
                   f"evaluacion_{dt.datetime.now():%Y%m%d_%H%M}.csv", "text/csv", use_container_width=True)
if c2.button("Guardar evaluación en resultados_en_vivo/", disabled=m["n"] == 0):
    carpeta_res = RAIZ / "resultados_en_vivo"
    carpeta_res.mkdir(exist_ok=True)
    sello = f"{dt.datetime.now():%Y%m%d_%H%M%S}"
    tabla.to_csv(carpeta_res / f"evaluacion_{sello}.csv", index=False)
    resumen = {"fecha": sello, "origen": st.session_state.origen, "imagenes": N, "etiquetadas": m["n"],
               "umbral": umbral, "tta": tta, "matriz": {"tp": m["tp"], "fn": m["fn"], "fp": m["fp"], "tn": m["tn"]},
               "accuracy": m["acc"], "precision": m["prec"], "recall": m["rec"], "f1": m["f1"],
               "cv": cv, "modelo": info.get("descripcion", info["archivo"])}
    (carpeta_res / f"evaluacion_{sello}.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False),
                                                          encoding="utf-8")
    st.success(f"Guardado en resultados_en_vivo/evaluacion_{sello}.csv y .json")

atajos_teclado()
