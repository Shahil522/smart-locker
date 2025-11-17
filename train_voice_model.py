#!/usr/bin/env python
"""
Enhanced voice model trainer with better preprocessing and augmentation.
Improves voice verification accuracy by:
1. Better noise filtering (spectral gating)
2. Audio normalization (peak + RMS)
3. Extended feature set (MFCC + Delta + Acceleration + 3rd derivative)
4. Weighted averaging (prioritize longer recordings)
"""
import os
import numpy as np
import pickle
import librosa
from pathlib import Path
from scipy import signal
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(__file__)
AUDIO_TRAIN_DIR = os.path.join(BASE_DIR, "static", "training_data", "user1")
MODEL_DIR = os.path.join(BASE_DIR, "models", "voice_models")
EMB_PATH = os.path.join(MODEL_DIR, "user1_mfcc.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

def denoise_audio(y, sr):
    """Remove background noise using spectral gating."""
    S = librosa.stft(y)
    S_db = librosa.power_to_db(np.abs(S) ** 2)
    
    # Estimate noise from quietest frames
    noise_threshold = np.percentile(S_db, 20)
    S_db = np.maximum(S_db, noise_threshold)
    
    # Convert back to linear scale
    S_filtered = librosa.db_to_power(S_db) ** 0.5
    y_filtered = librosa.istft(S_filtered)
    
    return y_filtered

def normalize_audio(y):
    """Normalize audio to standard loudness (peak + RMS)."""
    # Peak normalization
    y = y / (np.max(np.abs(y)) + 1e-8)
    
    # RMS normalization (consistent loudness)
    target_rms = 0.1
    current_rms = np.sqrt(np.mean(y ** 2))
    if current_rms > 1e-8:
        y = y * (target_rms / current_rms)
    
    return y

def extract_mfcc_embedding_enhanced(wav_path):
    """Extract enhanced MFCC features with preprocessing."""
    try:
        # Load audio
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        
        # Remove silence
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # Denoise
        y = denoise_audio(y, sr)
        
        # Normalize
        y = normalize_audio(y)
        
        # Standardize length
        if len(y) < 16000:
            y = np.pad(y, (0, max(0, 16000 - len(y))), mode='constant')
        else:
            y = y[:16000]
        
        # Extract MFCC with more coefficients
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        d1 = librosa.feature.delta(mfcc)
        d2 = librosa.feature.delta(mfcc, order=2)
        d3 = librosa.feature.delta(mfcc, order=3)
        
        # Extended feature set for better discrimination
        feat = np.concatenate([
            np.mean(mfcc, axis=1),      # MFCC mean
            np.std(mfcc, axis=1),       # MFCC std
            np.mean(d1, axis=1),        # Delta mean
            np.std(d1, axis=1),         # Delta std
            np.mean(d2, axis=1),        # Acceleration mean
            np.std(d2, axis=1),         # Acceleration std
            np.mean(d3, axis=1),        # 3rd derivative mean
            np.std(d3, axis=1),         # 3rd derivative std
        ])
        
        # L2 normalization for cosine similarity
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        
        return feat, len(y)  # Return length as weight
        
    except Exception as e:
        print(f"Error extracting features from {wav_path}: {e}")
        return None, 0

# Find all augmented voice samples
if not os.path.exists(AUDIO_TRAIN_DIR):
    print(f"ERROR: Training directory not found: {AUDIO_TRAIN_DIR}")
    exit(1)

wav_files = sorted([f for f in os.listdir(AUDIO_TRAIN_DIR) if f.endswith('.wav')])
print(f"Found {len(wav_files)} voice samples in {AUDIO_TRAIN_DIR}")

if not wav_files:
    print("ERROR: No WAV files found in training directory!")
    exit(1)

# Extract embeddings with weighting
embeddings = []
weights = []
valid_count = 0

print("\nProcessing voice samples...")
for idx, wav_file in enumerate(wav_files, 1):
    wav_path = os.path.join(AUDIO_TRAIN_DIR, wav_file)
    emb, length = extract_mfcc_embedding_enhanced(wav_path)
    
    if emb is not None:
        embeddings.append(emb)
        # Weight by audio length (longer = more trustworthy)
        weight = max(0.5, min(2.0, length / 16000))
        weights.append(weight)
        valid_count += 1
        
        if idx % 20 == 0:
            print(f"  {idx}/{len(wav_files)} samples processed...")

print(f"\n✓ Extracted {valid_count} valid embeddings")

if not embeddings:
    print("ERROR: No valid embeddings extracted!")
    exit(1)

# Create weighted average (longer samples get more influence)
embeddings_array = np.stack(embeddings, axis=0)
weights_array = np.array(weights)
weights_array = weights_array / np.sum(weights_array)  # Normalize weights

# Weighted average
avg_embedding = np.average(embeddings_array, axis=0, weights=weights_array)

# Additional normalization
avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)

print(f"✓ Weighted average embedding shape: {avg_embedding.shape}")
print(f"✓ Average weight per sample: {np.mean(weights_array):.3f}")

# Calculate embedding statistics for debug
similarities = []
for emb in embeddings_array:
    sim = cosine_similarity([avg_embedding], [emb])[0][0]
    similarities.append(sim)

similarities = np.array(similarities)
print(f"✓ Similarity statistics within training set:")
print(f"  - Mean: {np.mean(similarities):.4f}")
print(f"  - Std:  {np.std(similarities):.4f}")
print(f"  - Min:  {np.min(similarities):.4f}")
print(f"  - Max:  {np.max(similarities):.4f}")

# Save model
with open(EMB_PATH, 'wb') as f:
    pickle.dump(avg_embedding, f)

print(f"\n✓ Enhanced voice model saved to: {EMB_PATH}")
print("\n✅ Voice training complete!")
print("   Your voice is now more robust to different recording conditions.")
print(f"\nRecommended VOICE_THRESHOLD: {np.mean(similarities) - np.std(similarities):.2f}")
