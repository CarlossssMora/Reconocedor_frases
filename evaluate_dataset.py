import os
import argparse
import pickle
import librosa
import numpy as np

from scipy.spatial.distance import cdist
from preprocessing import preprocess_audio
from mfcc import extract_mfcc

# =========================
# ARGUMENTOS
# =========================
parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    required=True,
    help="Carpeta del dataset a evaluar"
)

parser.add_argument(
    "--model",
    required=True,
    help="Ruta del modelo .pkl"
)

args = parser.parse_args()

dataset_path = args.dataset
model_path = args.model

# =========================
# CARGAR MODELO
# =========================
with open(model_path, "rb") as f:
    modelos = pickle.load(f)

# =========================
# VARIABLES DE EVALUACION
# =========================
total = 0
correctos = 0
resultados_por_clase = {}

# =========================
# EVALUACION
# =========================
for clase_real in sorted(os.listdir(dataset_path)):

    clase_path = os.path.join(dataset_path, clase_real)

    if not os.path.isdir(clase_path):
        continue

    resultados_por_clase[clase_real] = {
        "total": 0,
        "correctos": 0
    }

    for archivo in sorted(os.listdir(clase_path)):

        if not archivo.lower().endswith(".wav"):
            continue

        ruta_audio = os.path.join(clase_path, archivo)

        try:
            audio, fs = librosa.load(
                ruta_audio,
                sr=16000
            )

            audio = preprocess_audio(audio, fs)
            mfccs = extract_mfcc(audio, fs)

            mejor_clase = None
            mejor_distancia = float("inf")

            for clase_modelo, codebook in modelos.items():

                distancias = cdist(
                    mfccs.T,
                    codebook,
                    metric="euclidean"
                )

                distancia_promedio = np.mean(
                    np.min(distancias, axis=1)
                )

                if distancia_promedio < mejor_distancia:
                    mejor_distancia = distancia_promedio
                    mejor_clase = clase_modelo

            total += 1
            resultados_por_clase[clase_real]["total"] += 1

            if mejor_clase == clase_real:
                correctos += 1
                resultados_por_clase[clase_real]["correctos"] += 1
            else:
                print(f"Error: {archivo}")
                print(f"  Real: {clase_real}")
                print(f"  Predicho: {mejor_clase}")
                print(f"  Distancia: {mejor_distancia:.4f}")

        except Exception as e:
            print(f"No se pudo procesar {ruta_audio}: {e}")

# =========================
# RESULTADOS
# =========================
print("\n==============================")
print("RESULTADOS GENERALES")
print("==============================")

if total > 0:
    accuracy = correctos / total * 100
else:
    accuracy = 0

print(f"Modelo evaluado: {model_path}")
print(f"Dataset evaluado: {dataset_path}")
print(f"Total de audios: {total}")
print(f"Correctos: {correctos}")
print(f"Exactitud: {accuracy:.2f}%")

print("\n==============================")
print("RESULTADOS POR CLASE")
print("==============================")

for clase, datos in resultados_por_clase.items():

    total_clase = datos["total"]
    correctos_clase = datos["correctos"]

    if total_clase > 0:
        acc_clase = correctos_clase / total_clase * 100
    else:
        acc_clase = 0

    print(f"{clase}: {correctos_clase}/{total_clase} = {acc_clase:.2f}%")