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

    return mfccs