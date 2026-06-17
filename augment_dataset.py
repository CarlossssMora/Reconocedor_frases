import os
import shutil
import librosa
import soundfile as sf
import numpy as np
import random

# =========================
# DATASETS DE PERSONAS
# =========================
datasets_personas = [
    "dataset_mike",
    "dataset_angel",
    "dataset_checo",
    "dataset_juan",
    "dataset_memo"
]

# =========================
# CARPETAS DE SALIDA
# =========================
dataset_general = "dataset_general"
dataset_mixto = "dataset_general_ruido" # Lo mantenemos con este nombre para que no cambies tus otros scripts

# =========================
# RUIDOS
# =========================
ruidos = {
    "air_conditioner": "ruido/air_conditioner.wav",
    "car_horn": "ruido/car_horn.wav",
    "children_playing": "ruido/children_playing.wav",
    "dog_bark": "ruido/dog_bark.wav",
    "street_music": "ruido/street_music.wav"
}

# Intensidad del ruido
alpha = 0.03

# =========================
# FUNCION PARA MEZCLAR AUDIO Y RUIDO
# =========================
def mix_audio(audio, ruido, alpha=0.03):
    if len(ruido) < len(audio):
        repeticiones = int(np.ceil(len(audio) / len(ruido)))
        ruido = np.tile(ruido, repeticiones)

    ruido = ruido[:len(audio)]

    max_ruido = np.max(np.abs(ruido))
    if max_ruido > 0:
        ruido = ruido / max_ruido

    audio_mezclado = audio + alpha * ruido
    audio_mezclado = np.clip(audio_mezclado, -1.0, 1.0)

    return audio_mezclado

# =========================
# VALIDACIONES
# =========================
for dataset in datasets_personas:
    if not os.path.exists(dataset):
        raise FileNotFoundError(f"No existe la carpeta: {dataset}")

for nombre_ruido, ruta_ruido in ruidos.items():
    if not os.path.exists(ruta_ruido):
        raise FileNotFoundError(f"No se encontró el ruido: {ruta_ruido}")

# =========================
# CREAR DATASET GENERAL LIMPIO
# =========================
if os.path.exists(dataset_general):
    shutil.rmtree(dataset_general)

os.makedirs(dataset_general, exist_ok=True)
print("Creando dataset general limpio...")

for dataset_persona in datasets_personas:
    nombre_persona = dataset_persona.replace("dataset_", "")
    for clase in os.listdir(dataset_persona):
        ruta_clase_origen = os.path.join(dataset_persona, clase)
        if not os.path.isdir(ruta_clase_origen): continue

        ruta_clase_destino = os.path.join(dataset_general, clase)
        os.makedirs(ruta_clase_destino, exist_ok=True)

        for archivo in os.listdir(ruta_clase_origen):
            if not archivo.lower().endswith(".wav"): continue

            ruta_origen = os.path.join(ruta_clase_origen, archivo)
            nuevo_nombre = f"{nombre_persona}_{archivo}"
            ruta_destino = os.path.join(ruta_clase_destino, nuevo_nombre)

            shutil.copy2(ruta_origen, ruta_destino)

print("Dataset general limpio creado correctamente.")

# =========================
# CREAR DATASET MIXTO EQUILIBRADO (Limpio + 1 Ruido Aleatorio)
# =========================
if os.path.exists(dataset_mixto):
    shutil.rmtree(dataset_mixto)

# Copiamos primero todos los audios limpios (500 archivos)
shutil.copytree(dataset_general, dataset_mixto)
print("\nGenerando audios con ruido (Relación 1:1)...")

# Convertimos el diccionario a una lista para poder usar random.choice
lista_ruidos = list(ruidos.items())

for clase in os.listdir(dataset_general):
    ruta_clase_limpia = os.path.join(dataset_general, clase)
    ruta_clase_ruido = os.path.join(dataset_mixto, clase)

    if not os.path.isdir(ruta_clase_limpia): continue

    for archivo in os.listdir(ruta_clase_limpia):
        if not archivo.lower().endswith(".wav"): continue

        ruta_audio = os.path.join(ruta_clase_limpia, archivo)
        audio, fs = librosa.load(ruta_audio, sr=16000)

        # ELEGIR UN RUIDO AL AZAR
        nombre_ruido, ruta_ruido = random.choice(lista_ruidos)

        ruido, _ = librosa.load(ruta_ruido, sr=16000)
        audio_con_ruido = mix_audio(audio, ruido, alpha=alpha)

        nombre_salida = archivo.replace(".wav", f"_noise_{nombre_ruido}.wav")
        ruta_salida = os.path.join(ruta_clase_ruido, nombre_salida)

        sf.write(ruta_salida, audio_con_ruido, fs)

print("\nAugmentación terminada correctamente.")
print(f"Dataset limpio: {dataset_general} (500 audios)")
print(f"Dataset mixto balanceado: {dataset_mixto} (1000 audios: 500 limpios + 500 ruidosos)")