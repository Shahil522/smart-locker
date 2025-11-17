# Full Final app.py — Smart Locker (Resemblyzer voice, OTP, face, training, verification)
# Requirements (install into your .venv):
#   pip install resemblyzer soundfile librosa numpy scikit-learn twilio vosk flask opencv-contrib-python
# Restart server after replacing this file.

import sys
# keep the fake webrtcvad shim used elsewhere (your project included fake_webrtcvad)
sys.modules['webrtcvad'] = __import__('fake_webrtcvad')

from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from datetime import datetime
import os, json, time, re, shutil, subprocess
import numpy as np
import pickle
import librosa
import soundfile as sf
import cv2
import difflib
import string
import wave
from sklearn.metrics.pairwise import cosine_similarity

# Twilio
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

# Try Resemblyzer (preferred)
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    RESEMBLYZER_OK = True
    encoder = VoiceEncoder()
except Exception as e:
    RESEMBLYZER_OK = False
    encoder = None
    print("⚠️ Resemblyzer import failed:", e)

# Vosk (optional) for phrase detection
VOSK_IMPORT_OK = True
try:
    from vosk import Model as VoskModel, KaldiRecognizer
except Exception:
    VOSK_IMPORT_OK = False
    VoskModel = KaldiRecognizer = None

# ---------------- Flask app ----------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("FLASK_SECRET", "super_secret_key_123")

# ---------------- Twilio config (set these) ----------------
# Replace with your Twilio credentials or keep as env vars
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
VERIFY_SERVICE_SID = os.environ.get("VERIFY_SERVICE_SID")
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# ---------------- Paths & constants ----------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SNAP_DIR = os.path.join(BASE_DIR, "static", "snapshots")
AUDIO_DIR = os.path.join(BASE_DIR, "static", "audio")
TRAINER_PATH = os.path.join(BASE_DIR, "face_training", "trainer", "trainer.yml")
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
MODEL_DIR = os.path.join(BASE_DIR, "models", "voice_models")
EMB_PATH = os.path.join(MODEL_DIR, "user1_resembly.pkl")

POSSIBLE_VOSK_DIRS = [
    os.path.join(BASE_DIR, "vosk_model"),
    os.path.join(BASE_DIR, "vosk-model-small-en-us-0.15"),
    os.path.join(BASE_DIR, "vosk-model-en-us-0.22")
]

os.makedirs(SNAP_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Security settings
UNLOCK_PHRASE = "hello sesame"
VOICE_THRESHOLD = 0.78       # Option A (chosen): high security but accepts your measured voice score
KEYWORD_SIM_THRESHOLD = 0.50
REQUIRE_BOTH = True

# OTP rate limiting (simple)
otp_rate = {}
MIN_INTERVAL_SECONDS = 30
MAX_PER_HOUR = 5

# ---------------- Vosk load (optional) ----------------
vosk_model = None
if VOSK_IMPORT_OK:
    for d in POSSIBLE_VOSK_DIRS:
        if os.path.exists(d):
            try:
                print("Loading Vosk model from:", d)
                vosk_model = VoskModel(d)
                print("✅ Vosk loaded.")
                break
            except Exception as e:
                print("Failed to load Vosk from:", d, e)
    if vosk_model is None:
        print("⚠️ Vosk installed but no usable model found.")
else:
    print("⚠️ Vosk not installed; phrase detection disabled (voice-only verification will be used).")

# ---------------- utilities ----------------
def append_log(entry):
    LOG_FILE = os.path.join(BASE_DIR, "access_log.json")
    try:
        data = json.load(open(LOG_FILE)) if os.path.exists(LOG_FILE) else {"logs": []}
    except Exception:
        data = {"logs": []}
    entry.setdefault("time", datetime.now().strftime("%Y-%m-%d %I:%M %p"))
    data["logs"].insert(0, entry)
    try:
        with open(LOG_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def normalize_phone(raw_phone: str, default_cc="+91"):
    raw = (raw_phone or "").strip()
    if raw.startswith('+') and re.fullmatch(r'\+\d{8,15}', raw):
        return raw
    digits = re.sub(r'\D', '', raw)
    if len(digits) == 10:
        return default_cc + digits
    if 11 <= len(digits) <= 15:
        return "+" + digits
    raise ValueError("Invalid phone number format")

def can_send_otp(phone: str):
    now = time.time()
    entry = otp_rate.get(phone, {"last_sent": 0, "hour_window": []})
    if now - entry["last_sent"] < MIN_INTERVAL_SECONDS:
        return False, int(MIN_INTERVAL_SECONDS - (now - entry["last_sent"]))
    window = [t for t in entry["hour_window"] if now - t <= 3600]
    if len(window) >= MAX_PER_HOUR:
        return False, "hour_limit"
    window.append(now)
    otp_rate[phone] = {"last_sent": now, "hour_window": window}
    return True, "ok"

def normalize_text(s):
    if not s: return ""
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s)
    return s

# ---------------- pages ----------------
@app.route("/")
def visitor_page():
    return render_template("visitor.html")

@app.route("/otp")
def otp_page():
    return render_template("otp_request.html")

@app.route("/face_recognition")
def face_recognition_page():
    return render_template("face_recognition.html")

@app.route("/voice_recognition")
def voice_recognition_page():
    return render_template("voice_recognition.html")

@app.route("/voice_training")
def voice_training_page():
    return render_template("voice_training.html")

# In-memory training samples for interactive sessions
TRAINING_SAMPLES = []

# ---------------- OTP endpoints (matching your frontend) ----------------
@app.route("/api/request_otp", methods=["POST"])
def request_otp():
    try:
        data = request.get_json() or {}
        raw_phone = data.get("phone", "")
        try:
            phone = normalize_phone(raw_phone)
        except ValueError:
            return jsonify({"message": "Invalid phone number"}), 400

        allowed, info = can_send_otp(phone)
        if not allowed:
            if info == "hour_limit":
                return jsonify({"message":"OTP limit reached for this hour"}), 429
            return jsonify({"message": f"Wait {info} seconds before retrying"}), 429

        # Use Twilio Verify (v2)
        try:
            twilio_client.verify.v2.services(VERIFY_SERVICE_SID).verifications.create(to=phone, channel="sms")
            append_log({"event": f"OTP sent to {phone}"})
            return jsonify({"message":"OTP sent successfully"})
        except TwilioRestException as e:
            append_log({"event": f"Twilio error: {e}"})
            return jsonify({"message": f"Twilio error: {str(e)}"}), 500
    except Exception as e:
        append_log({"event": f"request_otp_error: {e}"})
        return jsonify({"message": f"Server error: {str(e)}"}), 500

@app.route("/api/verify_otp", methods=["POST"])
def verify_otp():
    try:
        data = request.get_json() or {}
        raw_phone = data.get("phone", "")
        code = data.get("otp", "")
        try:
            phone = normalize_phone(raw_phone)
        except ValueError:
            return jsonify({"result":"error","message":"Invalid phone"}), 400
        if not code:
            return jsonify({"result":"error","message":"OTP required"}), 400
        try:
            check = twilio_client.verify.v2.services(VERIFY_SERVICE_SID).verification_checks.create(to=phone, code=code)
            if check.status == "approved":
                session["otp_verified"] = True
                append_log({"event": f"OTP approved for {phone}"})
                return jsonify({"result":"success"})
            else:
                return jsonify({"result":"failed"})
        except TwilioRestException as e:
            append_log({"event": f"Twilio verify error: {e}"})
            return jsonify({"result":"error","message":str(e)}), 500
    except Exception as e:
        append_log({"event": f"verify_otp_error: {e}"})
        return jsonify({"result":"error","message":str(e)}), 500

# ---------------- Face recognition ----------------
@app.route("/api/verify_face", methods=["POST"])
def verify_face():
    img = request.data
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"verify_{ts}.jpg"
    fpath = os.path.join(SNAP_DIR, fname)
    with open(fpath, "wb") as f:
        f.write(img)

    if not os.path.exists(TRAINER_PATH):
        return jsonify({"verified": False, "message": "No trained face model."})

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(TRAINER_PATH)
        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

        frame = cv2.imread(fpath)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            if confidence < 65:
                session["face_verified"] = True
                append_log({"event":"face_verified"})
                return jsonify({"verified": True, "confidence": float(confidence)})
        return jsonify({"verified": False, "message":"face not recognized"})
    except Exception as e:
        append_log({"event": f"face error: {e}"})
        return jsonify({"verified": False, "message": str(e)}), 500

# ---------------- audio helpers ----------------
def ensure_wav(input_path):
    """Convert to mono 16k WAV in-place (librosa or ffmpeg fallback)."""
    try:
        y, sr = librosa.load(input_path, sr=16000, mono=True)
        sf.write(input_path, y, 16000)
    except Exception:
        ffmpeg = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
        if ffmpeg:
            tmp = input_path + ".wav"
            cmd = [ffmpeg, "-y", "-i", input_path, "-ar", "16000", "-ac", "1", tmp]
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                if os.path.exists(tmp):
                    os.replace(tmp, input_path)
            except Exception:
                pass
    return input_path

# ---------------- Resemblyzer-based extractor ----------------
def extract_voice_embedding(path):
    """Return a normalized embedding vector using Resemblyzer."""
    if not RESEMBLYZER_OK:
        append_log({"event":"resemblyzer_not_available"})
        return None
    try:
        # preprocess_wav handles loading/resampling; returns numpy float32 waveform
        wav = preprocess_wav(path)
        emb = encoder.embed_utterance(wav)
        emb = np.asarray(emb, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / (norm + 1e-8)
        return emb
    except Exception as e:
        append_log({"event": f"embedding_error: {e}"})
        return None

def run_vosk_transcription(wav_path):
    if vosk_model is None:
        return ""
    ensure_wav(wav_path)
    try:
        wf = wave.open(wav_path, "rb")
    except Exception:
        return ""
    rec = KaldiRecognizer(vosk_model, wf.getframerate())
    rec.SetWords(False)
    text = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            try:
                text += " " + json.loads(rec.Result()).get("text","")
            except Exception:
                pass
    try:
        text += " " + json.loads(rec.FinalResult()).get("text","")
    except Exception:
        pass
    try: wf.close()
    except: pass
    return text.strip().lower()

# ---------------- Training endpoints ----------------
@app.route("/api/auto_train", methods=["POST"])
def auto_train():
    """
    Accepts multiple form files under key 'audio' and averages Resemblyzer embeddings.
    Client should upload 5 samples in one request (auto-train workflow).
    """
    files = request.files.getlist("audio")
    if not files:
        return jsonify({"status":"error","message":"No audio files sent"}), 400

    embeddings = []
    try:
        for f in files:
            tmp = os.path.join(AUDIO_DIR, f"train_tmp_{int(time.time()*1000)}.wav")
            f.save(tmp)
            ensure_wav(tmp)
            emb = extract_voice_embedding(tmp)
            if emb is not None:
                embeddings.append(emb)
            try: os.remove(tmp)
            except: pass

        if not embeddings:
            return jsonify({"status":"error","message":"No embeddings extracted"}), 500

        avg = np.mean(np.stack(embeddings, axis=0), axis=0)
        if np.linalg.norm(avg) > 0:
            avg = avg / np.linalg.norm(avg)

        with open(EMB_PATH, "wb") as fh:
            pickle.dump({"emb": avg, "type": "resemblyzer"}, fh)

        append_log({"event": f"Auto-trained {len(embeddings)} samples", "type":"resemblyzer"})
        return jsonify({"status":"ok","samples":len(embeddings), "type":"resemblyzer"})
    except Exception as e:
        append_log({"event": f"auto_train_error: {e}"})
        return jsonify({"status":"error","message":str(e)}), 500

@app.route("/api/train_one", methods=["POST"])
def train_one():
    """Train from a single sample (fallback)."""
    if "audio" not in request.files:
        return jsonify({"status":"error","message":"No audio"}), 400
    f = request.files["audio"]
    save_path = os.path.join(AUDIO_DIR, "user1.wav")
    f.save(save_path)
    ensure_wav(save_path)
    emb = extract_voice_embedding(save_path)
    if emb is None:
        try: os.remove(save_path)
        except: pass
        return jsonify({"status":"error","message":"feature extraction failed"}), 500
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    with open(EMB_PATH, "wb") as fh:
        pickle.dump({"emb": emb, "type": "resemblyzer"}, fh)
    append_log({"event":"single_train", "type":"resemblyzer"})
    return jsonify({"status":"ok", "type":"resemblyzer"})

# Interactive training endpoints
@app.route("/api/train_voice", methods=["POST"])
def train_voice():
    global TRAINING_SAMPLES
    if "audio" not in request.files:
        return jsonify({"status": "error", "message": "No audio provided"}), 400
    f = request.files["audio"]
    timestamp = int(time.time() * 1000)
    save_path = os.path.join(AUDIO_DIR, f"training_{timestamp}.wav")
    f.save(save_path)
    ensure_wav(save_path)
    emb = extract_voice_embedding(save_path)
    if emb is not None:
        TRAINING_SAMPLES.append((save_path, emb))
        return jsonify({"status": "ok", "samples": len(TRAINING_SAMPLES)})
    else:
        try: os.remove(save_path)
        except: pass
        return jsonify({"status":"error","message":"Could not extract features"}), 500

@app.route("/api/finalize_voice_training", methods=["POST"])
def finalize_voice_training():
    global TRAINING_SAMPLES
    if len(TRAINING_SAMPLES) < 1:
        return jsonify({"status":"error","message":"No training samples"}), 400
    try:
        embeddings = []
        weights = []
        for save_path, emb in TRAINING_SAMPLES:
            embeddings.append(emb)
            try:
                file_size = os.path.getsize(save_path)
            except Exception:
                file_size = 48000
            weight = max(0.5, min(2.0, file_size / 48000))
            weights.append(weight)
        embeddings_array = np.stack(embeddings, axis=0)
        weights_array = np.array(weights)
        weights_array = weights_array / np.sum(weights_array)
        avg_embedding = np.average(embeddings_array, axis=0, weights=weights_array)
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
        with open(EMB_PATH, 'wb') as f:
            pickle.dump({"emb": avg_embedding, "type": "resemblyzer"}, f)
        similarities = []
        for emb in embeddings_array:
            sim = float(cosine_similarity([avg_embedding], [emb])[0][0])
            similarities.append(sim)
        similarities = np.array(similarities)
        mean_sim = float(np.mean(similarities))
        std_sim = float(np.std(similarities))
        recommended_threshold = float(max(0.75, mean_sim - (std_sim * 1.5)))
        append_log({
            "event": "voice_training_complete",
            "samples": len(TRAINING_SAMPLES),
            "mean_similarity": mean_sim,
            "std_similarity": std_sim,
            "threshold_recommended": recommended_threshold,
            "type": "resemblyzer"
        })
        for save_path, _ in TRAINING_SAMPLES:
            try: os.remove(save_path)
            except: pass
        TRAINING_SAMPLES = []
        return jsonify({
            "status":"ok",
            "samples": len(embeddings),
            "mean_similarity": mean_sim,
            "std_similarity": std_sim,
            "recommended_threshold": recommended_threshold,
            "type": "resemblyzer"
        })
    except Exception as e:
        append_log({"event": f"voice_training_error: {e}"})
        return jsonify({"status":"error","message":str(e)}), 500

# ---------------- Verification endpoint (voice + phrase) ----------------
@app.route("/api/send_audio", methods=["POST"])
def send_audio():
    incoming_path = os.path.join(AUDIO_DIR, f"incoming_test_{int(time.time()*1000)}.wav")
    with open(incoming_path, "wb") as f:
        f.write(request.data)
    ensure_wav(incoming_path)

    # phrase detection
    text = ""
    keyword_ok = False
    if vosk_model is not None:
        text = run_vosk_transcription(incoming_path)
        tnorm = normalize_text(text)
        target = normalize_text(UNLOCK_PHRASE)
        sim = difflib.SequenceMatcher(None, tnorm, target).ratio()
        keyword_ok = (target in tnorm) or (sim >= KEYWORD_SIM_THRESHOLD)
    else:
        sim = 0.0
        text = ""

    voice_ok = False
    score = 0.0
    emb_type_used = None
    if os.path.exists(EMB_PATH):
        try:
            data = pickle.load(open(EMB_PATH, "rb"))
            emb_ref = data.get("emb") if isinstance(data, dict) else data
            emb_type_used = data.get("type", "resemblyzer") if isinstance(data, dict) else "resemblyzer"

            emb_new = extract_voice_embedding(incoming_path)
            if emb_new is None:
                append_log({"event":"voice_verify_error","message":"failed_extract_new"})
            else:
                emb_ref = np.array(emb_ref, dtype=np.float32)
                emb_new = np.array(emb_new, dtype=np.float32)
                if np.linalg.norm(emb_ref) > 0:
                    emb_ref = emb_ref / (np.linalg.norm(emb_ref) + 1e-8)
                if np.linalg.norm(emb_new) > 0:
                    emb_new = emb_new / (np.linalg.norm(emb_new) + 1e-8)

                score = float(cosine_similarity([emb_ref], [emb_new])[0][0])
                voice_ok = score >= VOICE_THRESHOLD

        except Exception as e:
            append_log({"event": f"voice_verify_error: {e}"})
    else:
        append_log({"event": "voice_verify_failed: no_trained_model"})

    # decide final
    if REQUIRE_BOTH and vosk_model is not None:
        verified = bool(keyword_ok and voice_ok)
        reason = "phrase+voice"
    else:
        verified = bool(voice_ok)
        reason = "voice-only" if not vosk_model else ("phrase+voice" if REQUIRE_BOTH else "either")

    append_log({
        "event": "verify_attempt",
        "text": text,
        "keyword_ok": bool(keyword_ok),
        "voice_ok": bool(voice_ok),
        "score": float(score),
        "verified": bool(verified),
        "reason": reason,
        "emb_type_used": emb_type_used
    })

    if verified:
        session["voice_verified"] = True

    # cleanup
    try:
        os.remove(incoming_path)
    except Exception:
        pass

    return jsonify({
        "verified": bool(verified),
        "keyword": bool(keyword_ok),
        "voice_match": bool(voice_ok),
        "text": text,
        "score": float(score),
        "sim": float(sim),
        "reason": reason,
        "emb_type_used": emb_type_used
    })

# ---------------- locker & logs ----------------
@app.route("/locker")
def locker_page():
    # require otp + face + voice (session flags)
    if not (session.get("otp_verified") and session.get("face_verified") and session.get("voice_verified")):
        return redirect(url_for("visitor_page"))
    return render_template("locker.html")

@app.route("/api/logs")
def get_logs():
    try:
        with open(os.path.join(BASE_DIR, "access_log.json"), "r") as f:
            return jsonify(json.load(f)["logs"])
    except Exception:
        return jsonify([])

# ---------------- RESET VOICE MODEL ----------------
@app.route("/api/reset_voice_model", methods=["POST"])
def reset_voice_model():
    try:
        if os.path.exists(EMB_PATH):
            os.remove(EMB_PATH)
        global TRAINING_SAMPLES
        for save_path, _ in TRAINING_SAMPLES:
            try: os.remove(save_path)
            except: pass
        TRAINING_SAMPLES = []
        append_log({"event": "voice_model_reset"})
        return jsonify({"status":"ok","message":"Old voice model deleted successfully. Ready for fresh training."})
    except Exception as e:
        append_log({"event": f"voice_model_reset_error: {e}"})
        return jsonify({"status":"error","message":str(e)}), 500

# ---------------- DEBUG: show similarity of saved model against training files ----------------
@app.route("/api/debug_sims")
def debug_sims():
    if not os.path.exists(EMB_PATH):
        return jsonify({"error":"no model"}), 404
    try:
        data = pickle.load(open(EMB_PATH,"rb"))
        ref = data.get("emb") if isinstance(data, dict) else data
        files = sorted([f for f in os.listdir(AUDIO_DIR) if f.startswith("training_") and f.endswith(".wav")])
        sims = []
        for fn in files:
            p = os.path.join(AUDIO_DIR, fn)
            emb = extract_voice_embedding(p)
            if emb is None:
                sims.append(None)
            else:
                sims.append(float(cosine_similarity([ref],[emb])[0][0]))
        return jsonify({"files":files, "sims":sims, "resemblyzer_ok": RESEMBLYZER_OK})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- run ----------------
if __name__ == "__main__":
    print("✅ Smart Locker server starting...")
    if RESEMBLYZER_OK:
        print("Resemblyzer support: AVAILABLE")
    else:
        print("Resemblyzer support: NOT AVAILABLE - voice features will fail.")
    if vosk_model is not None:
        print("Vosk support: AVAILABLE for phrase detection")
    else:
        print("Vosk support: NOT AVAILABLE - phrase detection disabled")

    # ✅ FIXED INDENTATION (4 spaces)
    app.run()
