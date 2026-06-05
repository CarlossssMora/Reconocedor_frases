import pickle
import sounddevice as sd
import numpy as np
import librosa

from scipy.spatial.distance import cdist
from preprocessing import preprocess_audio
from mfcc import extract_mfcc

# =========================
# CONFIGURACION
# =========================
FS = 16000
DURACION = 2

# =========================
# CARGAR MODELOS
# =========================
with open("modelos/codebooks.pkl", "rb") as f:
    modelos = pickle.load(f)

# =========================
# RECONOCIMIENTO EN VIVO
# =========================
while True:

    input("\nPresiona ENTER para grabar")

    print("Grabando...")

    audio = sd.rec(
        int(DURACION * FS),
        samplerate=FS,
        channels=1,
        dtype='float32'
    )

    sd.wait()

    audio = audio.flatten()

    audio = preprocess_audio(audio, FS)

    mfccs = extract_mfcc(audio, FS)

    mejor_palabra = None
    mejor_distancia = float("inf")

    for palabra, codebook in modelos.items():

        distancias = cdist(
            mfccs.T,
            codebook,
            metric='euclidean'
        )

        distancia_promedio = np.mean(
            np.min(distancias, axis=1)
        )

        if distancia_promedio < mejor_distancia:

            mejor_distancia = distancia_promedio
            mejor_palabra = palabra

    print("\n====================")
    print(f"Reconocido: {mejor_palabra}")
    print(f"Distancia: {mejor_distancia:.4f}")
    print("====================")