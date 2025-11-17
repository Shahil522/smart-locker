# voice_utils.py
import os
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import librosa

SAMPLE_RATE = 16000
DURATION_DEFAULT = 3
N_MFCC = 40

def record_audio(filename: str, duration: int = DURATION_DEFAULT, fs: int = SAMPLE_RATE):
    """
    Record from default mic and save WAV (16-bit PCM).
    Returns True on success.
    """
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    print(f"[voice_utils] Recording {duration}s -> {filename}")
    try:
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        scaled = np.int16(audio.flatten() * 32767)
        write(filename, fs, scaled)
        print("[voice_utils] Saved:", filename)
        return True
    except Exception as e:
        print("[voice_utils] Recording failed:", e)
        return False

def load_audio(path: str, sr: int = SAMPLE_RATE):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr

def extract_embedding(path: str, sr: int = SAMPLE_RATE, n_mfcc: int = N_MFCC):
    """
    Extract a normalized embedding vector from audio path.
    Using mean+std of MFCCs (2 * n_mfcc dims) then L2-normalize.
    """
    y, sr = load_audio(path, sr)
    # Pre-emphasis
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])
    # Trim leading/trailing silence
    y, _ = librosa.effects.trim(y, top_db=25)
    if len(y) < 0.5 * sr:
        pad_len = int(0.5 * sr) - len(y)
        y = np.pad(y, (0, pad_len), 'constant')
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)
    emb = np.concatenate([mean, std])
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb

def average_embeddings(embs):
    embs = np.vstack(embs)
    mean = np.mean(embs, axis=0)
    norm = np.linalg.norm(mean)
    if norm > 0:
        mean = mean / norm
    return mean
