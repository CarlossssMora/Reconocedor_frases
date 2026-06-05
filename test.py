import pickle
import librosa
import numpy as np

from preprocessing import preprocess_audio
from mfcc import extract_mfcc
from scipy.spatial.distance import cdist

# =========================
# CARGAR MODELOS
# =========================
with open("modelos/codebooks.pkl", "rb") as f:
    modelos = pickle.load(f)

# =========================
# AUDIO DE PRUEBA
# =========================
ruta = input("Ruta del audio: ")

audio, fs = librosa.load(
    ruta,
    sr=16000
)

audio = preprocess_audio(audio, fs)

mfccs = extract_mfcc(audio, fs)

# =========================
# RECONOCIMIENTO
# =========================
mejor_frase = None
mejor_distancia = float("inf")

for frase, codebook in modelos.items():

    distancias = cdist(
        mfccs.T,
        codebook,
        metric='euclidean'
    )

    distancia_promedio = np.mean(
        np.min(distancias, axis=1)
    )

    print(f"{frase}: {distancia_promedio:.4f}")

    if distancia_promedio < mejor_distancia:

        mejor_distancia = distancia_promedio
        mejor_frase = frase

print("\n===================")
print(f"Resultado: {mejor_frase}")
print("===================")