import os
import sounddevice as sd
from scipy.io.wavfile import write

# =========================
# CONFIGURACION
# =========================
FS = 16000          # Frecuencia de muestreo
DURACION = 4        # Duracion en segundos para frases

# =========================
# FRASES DEL DATASET
# Cada tupla tiene:
# ("frase que se dice", "nombre_de_carpeta")
# =========================
frases = [
    ("encender las luces de la sala", "encender_luces_sala"),
    ("apagar las luces de la sala", "apagar_luces_sala"),
    ("abrir la puerta principal", "abrir_puerta_principal"),
    ("cerrar la puerta principal", "cerrar_puerta_principal"),
    ("avanzar hacia adelante ahora", "avanzar_adelante_ahora"),
    ("retroceder hacia atras ahora", "retroceder_atras_ahora"),
    ("girar a la derecha lentamente", "girar_derecha_lentamente"),
    ("girar a la izquierda lentamente", "girar_izquierda_lentamente"),
    ("iniciar el sistema de control", "iniciar_sistema_control"),
    ("detener el sistema de control", "detener_sistema_control")
]

# =========================
# DATOS DEL PARTICIPANTE
# =========================
nombre = input("Nombre del participante: ").lower().replace(" ", "_")

# =========================
# GRABACION
# =========================
for frase, carpeta_nombre in frases:

    carpeta = os.path.join("dataset", carpeta_nombre)
    os.makedirs(carpeta, exist_ok=True)

    print(f"\n========== FRASE: {frase.upper()} ==========")

    for i in range(1, 11):

        input(f"\nPresiona ENTER para grabar audio {i}")

        print("Grabando...")

        audio = sd.rec(
            int(DURACION * FS),
            samplerate=FS,
            channels=1,
            dtype="float32"
        )

        sd.wait()

        nombre_archivo = f"{nombre}_{i:02d}.wav"
        ruta = os.path.join(carpeta, nombre_archivo)

        write(
            ruta,
            FS,
            (audio * 32767).astype("int16")
        )

        print(f"Guardado: {ruta}")

print("\nDataset grabado correctamente")