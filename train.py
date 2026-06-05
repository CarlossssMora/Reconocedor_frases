import os
import librosa
import pickle
import numpy as np

from preprocessing import preprocess_audio
from mfcc import extract_mfcc
from vq_codebook import generate_codebook

# =========================
# CONFIGURACION
# =========================
dataset_path = "dataset"
modelos = {}

# =========================
# ENTRENAMIENTO
# =========================
for palabra in os.listdir(dataset_path):

    palabra_path = os.path.join(dataset_path, palabra)

    if not os.path.isdir(palabra_path):
        continue

    print(f"\nEntrenando: {palabra}")

    all_mfcc = []

    for archivo in os.listdir(palabra_path):

        ruta = os.path.join(
            palabra_path,
            archivo
        )

        audio, fs = librosa.load(
            ruta,
            sr=16000
        )

        audio = preprocess_audio(audio, fs)

        mfccs = extract_mfcc(audio, fs)

        all_mfcc.append(mfccs)

    all_mfcc = np.hstack(all_mfcc)

    codebook = generate_codebook(
        all_mfcc,
        k=16
    )

    modelos[palabra] = codebook

# =========================
# GUARDAR MODELOS
# =========================
os.makedirs("modelos", exist_ok=True)

with open("modelos/codebooks.pkl", "wb") as f:
    pickle.dump(modelos, f)

print("\nEntrenamiento finalizado")