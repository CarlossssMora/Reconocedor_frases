import pickle
import sounddevice as sd
import numpy as np
import streamlit as st

from scipy.spatial.distance import cdist
from preprocessing import preprocess_audio
from mfcc import extract_mfcc

# =========================
# CONFIGURACION
# =========================
FS = 16000
DURACION = 4

# =========================
# DICCIONARIO DE FRASES
# llave = nombre de carpeta
# valor = frase original
# =========================
frases_originales = {
    "encender_luces_sala": "Encender las luces de la sala",
    "apagar_luces_sala": "Apagar las luces de la sala",
    "abrir_puerta_principal": "Abrir la puerta principal",
    "cerrar_puerta_principal": "Cerrar la puerta principal",
    "avanzar_adelante_ahora": "Avanzar hacia adelante ahora",
    "retroceder_atras_ahora": "Retroceder hacia atrás ahora",
    "girar_derecha_lentamente": "Girar a la derecha lentamente",
    "girar_izquierda_lentamente": "Girar a la izquierda lentamente",
    "iniciar_sistema_control": "Iniciar el sistema de control",
    "detener_sistema_control": "Detener el sistema de control"
}

# =========================
# INTERFAZ
# =========================
st.set_page_config(page_title="Reconocimiento de Frases", page_icon="🎙️")

st.title("Reconocimiento de frases en vivo")
st.write("Modelo basado en MFCC + VQ + distancia euclidiana")

# =========================
# FRASES DISPONIBLES
# =========================
st.subheader("Frases disponibles")

st.info(
    "\n".join([
        f"• {frase}\n"
        for frase in frases_originales.values()
    ])
)

# =========================
# CARGAR MODELOS
# =========================
@st.cache_resource
def cargar_modelos():
    with open("modelos/codebooks.pkl", "rb") as f:
        return pickle.load(f)

modelos = cargar_modelos()

st.success("Modelo cargado correctamente")

# =========================
# BOTON DE GRABACION
# =========================
if st.button("Grabar frase"):

    with st.spinner("Grabando audio..."):

        audio = sd.rec(
            int(DURACION * FS),
            samplerate=FS,
            channels=1,
            dtype="float32"
        )

        sd.wait()

    st.info("Procesando audio...")

    audio = audio.flatten()

    audio = preprocess_audio(audio, FS)

    mfccs = extract_mfcc(audio, FS)

    mejor_clase = None
    mejor_distancia = float("inf")

    resultados = {}

    for clase, codebook in modelos.items():

        distancias = cdist(
            mfccs.T,
            codebook,
            metric="euclidean"
        )

        distancia_promedio = np.mean(
            np.min(distancias, axis=1)
        )

        resultados[clase] = distancia_promedio

        if distancia_promedio < mejor_distancia:
            mejor_distancia = distancia_promedio
            mejor_clase = clase

    frase_detectada = frases_originales.get(
        mejor_clase,
        mejor_clase
    )

    st.subheader("Resultado")
    st.success(f"Frase reconocida: **{frase_detectada}**")
    st.write(f"Clase detectada: `{mejor_clase}`")
    st.write(f"Distancia: `{mejor_distancia:.4f}`")

    st.subheader("Distancias por clase")

    for clase, distancia in sorted(resultados.items(), key=lambda x: x[1]):
        frase_original = frases_originales.get(clase, clase)
        st.write(f"**{frase_original}**: {distancia:.4f}")