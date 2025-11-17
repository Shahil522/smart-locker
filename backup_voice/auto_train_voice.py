# auto_train_voice.py
import time
from train_voice import train_user
from voice_utils import record_audio, extract_embedding, average_embeddings
import os, pickle

# This script demonstrates collecting samples and updating model incrementally.
# Use cautiously on a server — it records from server mic.

MODEL_DIR = "models/voice_models"

def incremental_enroll(username, new_samples=2, duration=3):
    """
    Records `new_samples` new wavs into enroll_voice/username and recomputes averaged model.
    If no model exists, runs full train flow.
    """
    user_dir = os.path.join("enroll_voice", username)
    os.makedirs(user_dir, exist_ok=True)

    # record new samples
    existing_files = sorted([f for f in os.listdir(user_dir) if f.endswith(".wav")])
    start_idx = len(existing_files)
    for i in range(new_samples):
        wav_path = os.path.join(user_dir, f"{start_idx + i}.wav")
        ok = record_audio(wav_path, duration=duration)
        if not ok:
            print("[auto_train_voice] Recording failed.")
            return False

    # build embeddings from all available samples
    embs = []
    for f in sorted([f for f in os.listdir(user_dir) if f.endswith(".wav")]):
        emb = extract_embedding(os.path.join(user_dir, f))
        embs.append(emb)
    if not embs:
        print("[auto_train_voice] No samples found.")
        return False
    final_emb = average_embeddings(embs)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, f"{username}.pkl"), "wb") as m:
        pickle.dump(final_emb, m)
    print("[auto_train_voice] Updated model for", username)
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--new_samples", type=int, default=2)
    parser.add_argument("--duration", type=int, default=3)
    args = parser.parse_args()
    incremental_enroll(args.username, new_samples=args.new_samples, duration=args.duration)
