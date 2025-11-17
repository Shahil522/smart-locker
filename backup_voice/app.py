# backend/app.py
import sys
sys.modules['webrtcvad'] = __import__('fake_webrtcvad')
from flask import Flask, request, jsonify, render_template, send_from_directory, session, redirect, url_for
from datetime import datetime
import os, json, base64, random, time, re
import cv2, numpy as np
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

# Voice recognition
from resemblyzer import VoiceEncoder, preprocess_wav
import soundfile as sf
import librosa
from auto_train_voice import auto_train  # ✅ NEW: automatic voice trainer import

# ---------- Flask App ----------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "super_secret_key_123"

# ---------- Twilio ----------
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
VERIFY_SERVICE_SID = os.environ.get("VERIFY_SERVICE_SID")

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# ---------- Paths ----------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SNAP_DIR = os.path.join(BASE_DIR, "static", "snapshots")
AUDIO_DIR = os.path.join(BASE_DIR, "static", "audio")
TRAINER_PATH = os.path.join(BASE_DIR, "face_training", "trainer", "trainer.yml")
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
os.makedirs(SNAP_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

CODE_FILE = os.path.join(BASE_DIR, "code_data.json")
ALERT_FILE = os.path.join(BASE_DIR, "alert.json")
LOG_FILE = os.path.join(BASE_DIR, "access_log.json")
OWNER_FILE = os.path.join(BASE_DIR, "owner_data.json")

# ---------- OTP Rate Limiting ----------
otp_rate = {}
MIN_INTERVAL_SECONDS = 60
MAX_PER_HOUR = 5

# ---------- Voice Encoder Initialization ----------
encoder = VoiceEncoder()
USER_VOICE_PATH = os.path.join(AUDIO_DIR, "user1.wav")
user_embedding = None

if os.path.exists(USER_VOICE_PATH):
    y, sr = librosa.load(USER_VOICE_PATH, sr=16000, mono=True)
    sf.write(USER_VOICE_PATH, y, 16000)
    user_embedding = encoder.embed_utterance(preprocess_wav(USER_VOICE_PATH))
    print("✅ Loaded reference voice successfully")
else:
    print("⚠️ Reference voice not found. Please upload user1.wav first.")

# ---------- Utility ----------
def append_log(entry):
    try:
        data = json.load(open(LOG_FILE)) if os.path.exists(LOG_FILE) else {"logs": []}
    except json.JSONDecodeError:
        data = {"logs": []}
    data["logs"].insert(0, entry)
    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=2)

def normalize_phone(raw_phone: str, default_cc="+91"):
    raw = raw_phone.strip()
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
        return False, f"Wait {int(MIN_INTERVAL_SECONDS - (now - entry['last_sent']))}s before next OTP."
    window = [t for t in entry["hour_window"] if now - t <= 3600]
    if len(window) >= MAX_PER_HOUR:
        return False, "Max OTPs reached for this hour."
    window.append(now)
    otp_rate[phone] = {"last_sent": now, "hour_window": window}
    return True, "OK"

def all_verified():
    return session.get("face_verified") and session.get("otp_verified") and session.get("voice_verified")

# ---------- Pages ----------
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

# ---------- OTP ----------
@app.route("/api/request_otp", methods=["POST"])
def request_otp():
    data = request.get_json() or {}
    raw_phone = data.get("phone", "")
    try:
        phone = normalize_phone(raw_phone)
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400

    allowed, msg = can_send_otp(phone)
    if not allowed:
        return jsonify({"status": "error", "message": msg}), 429

    try:
        twilio_client.verify.services(VERIFY_SERVICE_SID).verifications.create(to=phone, channel="sms")
        append_log({"time": datetime.now().strftime("%Y-%m-%d %I:%M %p"), "event": f"OTP sent to {phone}"})
        return jsonify({"status": "ok", "message": f"OTP sent to {phone}"})
    except TwilioRestException as e:
        return jsonify({"status": "error", "message": f"Twilio error: {e}"}), 500

@app.route("/api/verify_otp", methods=["POST"])
def verify_otp():
    data = request.get_json() or {}
    phone = normalize_phone(data.get("phone", ""))
    code = data.get("otp", "")
    if not phone or not code:
        return jsonify({"status": "error", "message": "Phone and OTP required"}), 400
    try:
        check = twilio_client.verify.services(VERIFY_SERVICE_SID).verification_checks.create(to=phone, code=code)
        result = "success" if check.status == "approved" else "fail"
        if result == "success":
            session["otp_verified"] = True
    except Exception:
        result = "error"
    append_log({"time": datetime.now().strftime("%Y-%m-%d %I:%M %p"), "event": f"OTP verify for {phone}: {result}"})
    return jsonify({"result": result})

# ---------- Face Recognition ----------
@app.route("/api/verify_face", methods=["POST"])
def verify_face():
    img = request.data
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"verify_{ts}.jpg"
    fpath = os.path.join(SNAP_DIR, fname)
    with open(fpath, "wb") as f:
        f.write(img)

    message = "Face not recognized."
    recognized = False

    try:
        if not os.path.exists(TRAINER_PATH):
            return jsonify({"verified": False, "message": "No trained model found."})

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(TRAINER_PATH)
        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

        frame = cv2.imread(fpath)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            id_, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if confidence < 65:
                recognized = True
                session["face_verified"] = True
                message = f"✅ Face verified ({round(100 - confidence, 2)}% confidence)"
            else:
                message = f"❌ Face not recognized (confidence={round(confidence, 2)})"

    except Exception as e:
        message = f"Error: {e}"

    append_log({"time": datetime.now().strftime("%Y-%m-%d %I:%M %p"), "event": message})
    return jsonify({"verified": recognized, "message": message})

# ---------- Voice ----------
@app.route("/api/train_voice", methods=["POST"])
def train_voice():
    """Uploads base user voice and auto-trains new embeddings"""
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio provided"}), 400

        file = request.files["audio"]
        filepath = os.path.join(AUDIO_DIR, "user1.wav")
        file.save(filepath)

        # Auto-train model
        auto_train(filepath)

        # Reload the new embedding
        global user_embedding
        y, sr = librosa.load(filepath, sr=16000, mono=True)
        sf.write(filepath, y, 16000)
        user_embedding = encoder.embed_utterance(preprocess_wav(filepath))

        return jsonify({"status": "ok", "message": "Voice model trained successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/api/send_audio", methods=["POST"])
def send_audio():
    global user_embedding
    audio = request.data
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"voice_{ts}.wav"
    fpath = os.path.join(AUDIO_DIR, fname)
    with open(fpath, "wb") as f:
        f.write(audio)

    try:
        y, sr = librosa.load(fpath, sr=16000, mono=True)
        sf.write(fpath, y, 16000)
    except Exception as e:
        return jsonify({"verified": False, "message": f"Audio conversion failed: {e}"})

    verified = False
    message = "No reference voice found."
    try:
        if user_embedding is not None:
            embedding = encoder.embed_utterance(preprocess_wav(fpath))
            similarity = np.dot(user_embedding, embedding) / (
                np.linalg.norm(user_embedding) * np.linalg.norm(embedding)
            )
            print(f"[DEBUG] Similarity = {similarity:.2f}")
            if similarity > 0.65:
                session["voice_verified"] = True
                verified = True
                message = f"✅ Voice verified (similarity={similarity:.2f})"
            else:
                message = f"❌ Voice not recognized (similarity={similarity:.2f})"
        else:
            message = "⚠️ Reference voice missing. Train first."
    except Exception as e:
        message = f"Error: {e}"

    append_log({"time": datetime.now().strftime("%Y-%m-%d %I:%M %p"), "event": message})
    return jsonify({"verified": verified, "message": message})

# ---------- Locker ----------
@app.route("/locker")
def locker_page():
    if not all_verified():
        return redirect(url_for("visitor_page"))
    return render_template("locker.html")

# ---------- Logs ----------
@app.route("/api/logs")
def get_logs():
    try:
        with open(LOG_FILE, "r") as f:
            return jsonify(json.load(f)["logs"])
    except Exception:
        return jsonify([])

# ---------- Run ----------
if __name__ == "__main__":
    for file, default in [(LOG_FILE, {"logs": []}), (ALERT_FILE, {"pending": False})]:
        if not os.path.exists(file):
            with open(file, "w") as f:
                json.dump(default, f, indent=2)
    app.run(debug=True)
