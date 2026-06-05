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
for frase in os.listdir(dataset_path):

    frase_path = os.path.join(dataset_path, frase)

    if not os.path.isdir(frase_path):
        continue

    print(f"\nEntrenando: {frase}")

    all_mfcc = []

    for archivo in os.listdir(frase_path):

        ruta = os.path.join(
            frase_path,
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

    modelos[frase] = codebook

# =========================
# GUARDAR MODELOS
# =========================
os.makedirs("modelos", exist_ok=True)

with open("modelos/codebooks.pkl", "wb") as f:
    pickle.dump(modelos, f)

print("\nEntrenamiento finalizado")