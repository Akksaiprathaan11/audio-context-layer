from pathlib import Path
import librosa
import numpy as np

DEFAULT_SR = 16000

def load_audio(path: str | Path, sr: int = DEFAULT_SR):
    y, actual_sr = librosa.load(str(path), sr=sr, mono=True)
    if y.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y.astype(np.float32), sr

def extract_mel(y: np.ndarray, sr: int = DEFAULT_SR, n_mels: int = 128):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    return librosa.power_to_db(mel, ref=np.max)
