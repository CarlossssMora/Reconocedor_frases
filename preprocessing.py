import numpy as np
import librosa
from scipy.signal import butter, lfilter

# =========================
# FILTRO PASA BANDA
# =========================
def bandpass_filter(audio, fs, lowcut=300, highcut=3400, order=5):

    nyquist = 0.5 * fs

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype='band')

    filtered = lfilter(b, a, audio)

    return filtered

# =========================
# NORMALIZACION
# =========================
def normalize_audio(audio):

    max_value = np.max(np.abs(audio))

    if max_value > 0:
        audio = audio / max_value

    return audio

# =========================
# ELIMINACION DE SILENCIO
# =========================
def remove_silence(audio):

    audio_trimmed, _ = librosa.effects.trim(
        audio,
        top_db=20
    )

    return audio_trimmed

# =========================
# PREPROCESAMIENTO COMPLETO
# =========================
def preprocess_audio(audio, fs):

    audio = remove_silence(audio)

    audio = bandpass_filter(audio, fs)

    audio = normalize_audio(audio)

    return audio