import os
import argparse
import librosa
import pickle
import numpy as np

from preprocessing import preprocess_audio
from mfcc import extract_mfcc
from vq_codebook import generate_codebook

# =========================
# ARGUMENTOS
# =========================
parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    default="dataset",
    help="Carpeta del dataset a usar"
)

parser.add_argument(
    "--output",
    default="modelos/codebooks.pkl",
    help="Ruta donde se guardará el modelo"
)

parser.add_argument(
    "--k",
    type=int,
    default=16,
    help="Número de centroides para K-Means"
)

args = parser.parse_args()

dataset_path = args.dataset
output_path = args.output
k = args.k

modelos = {}

# =========================
# VALIDAR DATASET
# =========================
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"No existe la carpeta del dataset: {dataset_path}")

# =========================
# ENTRENAMIENTO
# =========================
for frase in sorted(os.listdir(dataset_path)):

    frase_path = os.path.join(dataset_path, frase)

    if not os.path.isdir(frase_path):
        continue

    print(f"\nEntrenando: {frase}")

    all_mfcc = []

    for archivo in sorted(os.listdir(frase_path)):

        if not archivo.lower().endswith(".wav"):
            continue

        ruta = os.path.join(frase_path, archivo)

        audio, fs = librosa.load(
            ruta,
            sr=16000
        )

        audio = preprocess_audio(audio, fs)

        mfccs = extract_mfcc(audio, fs)

        all_mfcc.append(mfccs)

    if len(all_mfcc) == 0:
        print(f"No hay audios en {frase}, se omite.")
        continue

    all_mfcc = np.hstack(all_mfcc)

    codebook = generate_codebook(
        all_mfcc,
        k=k
    )

    modelos[frase] = codebook

# =========================
# GUARDAR MODELO
# =========================
os.makedirs(
    os.path.dirname(output_path),
    exist_ok=True
)

with open(output_path, "wb") as f:
    pickle.dump(modelos, f)

print("\nEntrenamiento finalizado")
print(f"Modelo guardado en: {output_path}")