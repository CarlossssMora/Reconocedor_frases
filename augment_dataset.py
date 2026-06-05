import os
import random
import librosa
import soundfile as sf
import numpy as np

# =========================
# RUIDOS
# =========================
ruidos = [
    "ruido/air_conditioner.wav",
    "ruido/car_horn.wav",
    "ruido/children_playing.wav",
    "ruido/dog_bark.wav",
    "ruido/street_music.wav"
]

# =========================
# MEZCLAR AUDIO + RUIDO
# =========================
def mix_audio(audio, noise, alpha=0.02):

    if len(noise) < len(audio):

        repeat = int(np.ceil(len(audio) / len(noise)))

        noise = np.tile(noise, repeat)

    noise = noise[:len(audio)]

    mixed = audio + alpha * noise

    return mixed

# =========================
# RECORRER DATASET
# =========================
dataset_path = "dataset"

for palabra in os.listdir(dataset_path):

    palabra_path = os.path.join(dataset_path, palabra)

    if not os.path.isdir(palabra_path):
        continue

    for archivo in os.listdir(palabra_path):

        if "_noise" in archivo:
            continue

        ruta_audio = os.path.join(palabra_path, archivo)

        audio, fs = librosa.load(
            ruta_audio,
            sr=16000
        )

        for i in range(2):

            ruido_path = random.choice(ruidos)

            ruido, _ = librosa.load(
                ruido_path,
                sr=16000
            )

            mixed = mix_audio(audio, ruido)

            nombre = archivo.replace(
                ".wav",
                f"_noise{i+1}.wav"
            )

            salida = os.path.join(
                palabra_path,
                nombre
            )

            sf.write(salida, mixed, fs)

            print(f"Generado: {salida}")

print("\nAugmentacion terminada")