# train_voice.py
import os
import json
import time
from voice_utils import extract_embedding

TRAIN_DIR = "static/audio/train"
DB_PATH = "code_data.json"

os.makedirs(TRAIN_DIR, exist_ok=True)

def save_embedding(uid, embedding):
    try:
        with open(DB_PATH, "r") as f:
            db = json.load(f)
    except:
        db = {}

    if uid not in db:
        db[uid] = {"embeddings": []}

    db[uid]["embeddings"].append(embedding)

    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=4)

def train_from_file(uid, file):
    # Save uploaded audio
    filepath = os.path.join(TRAIN_DIR, f"train_{uid}_{int(time.time())}.wav")
    file.save(filepath)

    # Extract embedding
    emb = extract_embedding(filepath)

    # Save to DB
    save_embedding(uid, emb)

    return True
