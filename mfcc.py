import librosa
import numpy as np

def extract_mfcc(audio, fs):
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=fs,
        n_mfcc=13,
        n_fft=512,
        hop_length=160
    )

    delta = librosa.feature.delta(mfccs)
    delta_delta = librosa.feature.delta(mfccs, order=2)

    mfcc_completo = np.vstack([
        mfccs,
        delta,
        delta_delta
    ])

    return mfcc_completo