# auto_train_voice.py
"""
Auto-train voice embedding from an uploaded wav file.
Produces a robust averaged MFCC embedding and saves it as a .pkl file.
"""

import os
import pickle
import numpy as np
import librosa
import soundfile as sf

BASE_MODEL_DIR = "models/voice_models"
SR = 16000
N_MFCC = 40


# ---------- Simple augmentations ----------
def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def time_stretch(y, rate):
    try:
        return librosa.effects.time_stretch(y, rate)
    except:
        return y

def pitch_shift(y, sr, steps):
    try:
        return librosa.effects.pitch_shift(y, sr, steps)
    except:
        return y


# ---------- Embedding ----------
def extract_embedding_from_signal(y, sr=SR):
    if y.ndim > 1:
        y = librosa.to_mono(y)

    # Pre-emphasis
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])

    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=25)

    # Pad if short
    if len(y) < int(0.5 * sr):
        y = np.pad(y, (0, int(0.5 * sr) - len(y)), mode="constant")

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)

    emb = np.concatenate([mean, std])
    emb = emb / np.linalg.norm(emb)
    return emb


# ---------- Auto Trainer ----------
def auto_train(filepath: str, username="user1"):
    if not os.path.exists(filepath):
        raise FileNotFoundError("Input file not found: " + filepath)

    # Load audio
    y, _ = librosa.load(filepath, sr=SR, mono=True)
    sf.write(filepath, y, SR)  # save standardized

    signals = [
        y,
        add_noise(y, 0.004),
        add_noise(y, 0.008),
        time_stretch(y, 0.96),
        time_stretch(y, 1.04),
        pitch_shift(y, SR, 1),
        pitch_shift(y, SR, -1)
    ]

    embeddings = []
    for sig in signals:
        try:
            emb = extract_embedding_from_signal(sig, sr=SR)
            embeddings.append(emb)
        except:
            pass

    if not embeddings:
        raise Exception("No embeddings were created.")

    final_emb = np.mean(np.vstack(embeddings), axis=0)
    final_emb = final_emb / np.linalg.norm(final_emb)

    os.makedirs(BASE_MODEL_DIR, exist_ok=True)
    out_path = os.path.join(BASE_MODEL_DIR, f"{username}.pkl")

    with open(out_path, "wb") as f:
        pickle.dump(final_emb, f)

    return out_path
