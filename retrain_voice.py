#!/usr/bin/env python
"""
Retrain voice model from recent voice samples in static/audio folder.
Uses all voice_* WAV files recorded recently for better accuracy.
"""
import os
import numpy as np
import pickle
import librosa
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(__file__)
AUDIO_DIR = os.path.join(BASE_DIR, "static", "audio")
MODEL_DIR = os.path.join(BASE_DIR, "models", "voice_models")
EMB_PATH = os.path.join(MODEL_DIR, "user1_mfcc.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

def denoise_audio(y, sr):
    """Remove background noise using spectral gating."""
    S = librosa.stft(y)
    S_db = librosa.power_to_db(np.abs(S) ** 2)
    noise_threshold = np.percentile(S_db, 20)
    S_db = np.maximum(S_db, noise_threshold)
    S_filtered = librosa.db_to_power(S_db) ** 0.5
    y_filtered = librosa.istft(S_filtered)
    return y_filtered

def normalize_audio(y):
    """Normalize audio to standard loudness."""
    y = y / (np.max(np.abs(y)) + 1e-8)
    target_rms = 0.1
    current_rms = np.sqrt(np.mean(y ** 2))
    if current_rms > 1e-8:
        y = y * (target_rms / current_rms)
    return y

def extract_mfcc_embedding_enhanced(wav_path):
    """Extract enhanced MFCC features."""
    try:
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        y, _ = librosa.effects.trim(y, top_db=20)
        y = denoise_audio(y, sr)
        y = normalize_audio(y)
        
        if len(y) < 16000:
            y = np.pad(y, (0, max(0, 16000 - len(y))), mode='constant')
        else:
            y = y[:16000]
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        d1 = librosa.feature.delta(mfcc)
        d2 = librosa.feature.delta(mfcc, order=2)
        d3 = librosa.feature.delta(mfcc, order=3)
        
        feat = np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(d1, axis=1),
            np.std(d1, axis=1),
            np.mean(d2, axis=1),
            np.std(d2, axis=1),
            np.mean(d3, axis=1),
            np.std(d3, axis=1),
        ])
        
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        
        return feat, len(y)
        
    except Exception as e:
        print(f"✗ Error processing {os.path.basename(wav_path)}: {e}")
        return None, 0

# Find all voice WAV files in audio directory
if not os.path.exists(AUDIO_DIR):
    print(f"ERROR: Audio directory not found: {AUDIO_DIR}")
    exit(1)

wav_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.startswith('voice_') and f.endswith('.wav')])
print(f"\nFound {len(wav_files)} voice samples in {AUDIO_DIR}")

if not wav_files:
    print("ERROR: No voice samples found! Record some voice samples first.")
    exit(1)

# Extract embeddings with weighting
embeddings = []
weights = []
valid_count = 0

print("\nProcessing voice samples...")
for idx, wav_file in enumerate(wav_files, 1):
    wav_path = os.path.join(AUDIO_DIR, wav_file)
    emb, length = extract_mfcc_embedding_enhanced(wav_path)
    
    if emb is not None:
        embeddings.append(emb)
        # Weight by audio length
        weight = max(0.5, min(2.0, length / 16000))
        weights.append(weight)
        valid_count += 1
        
        if idx % 10 == 0 or idx == len(wav_files):
            print(f"  {idx}/{len(wav_files)} samples processed...")

print(f"\n✓ Extracted {valid_count} valid voice embeddings")

if not embeddings:
    print("ERROR: No valid embeddings extracted!")
    exit(1)

# Create weighted average
embeddings_array = np.stack(embeddings, axis=0)
weights_array = np.array(weights)
weights_array = weights_array / np.sum(weights_array)

avg_embedding = np.average(embeddings_array, axis=0, weights=weights_array)
avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)

print(f"✓ Weighted average embedding shape: {avg_embedding.shape}")

# Calculate statistics
similarities = []
for emb in embeddings_array:
    sim = cosine_similarity([avg_embedding], [emb])[0][0]
    similarities.append(sim)

similarities = np.array(similarities)
print(f"\n✓ Voice consistency statistics:")
print(f"  - Mean similarity:  {np.mean(similarities):.4f}")
print(f"  - Std deviation:    {np.std(similarities):.4f}")
print(f"  - Min:              {np.min(similarities):.4f}")
print(f"  - Max:              {np.max(similarities):.4f}")

# Determine optimal threshold
optimal_threshold = np.mean(similarities) - (np.std(similarities) * 1.5)
print(f"\n✓ Recommended VOICE_THRESHOLD: {optimal_threshold:.2f}")

# Save model
with open(EMB_PATH, 'wb') as f:
    pickle.dump(avg_embedding, f)

print(f"\n✓ Voice model saved to: {EMB_PATH}")
print("\n✅ Voice retraining complete!")
print(f"   Training samples: {valid_count}")
print(f"   Model threshold: {optimal_threshold:.2f}")
print("\n   Try voice verification again - it should work better now!")
