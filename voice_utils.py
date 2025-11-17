# Add these imports at top of voice_utils.py
import numpy as np
import soundfile as sf
import torch

# Add these new functions (append to file)
def _read_wav_for_speaker(wav_path, target_sr=16000):
    # returns numpy ndarray (1, n_samples) float32
    y, sr = sf.read(wav_path, dtype='float32')
    if y.ndim > 1:
        y = y.mean(axis=1)
    # resample if needed using librosa (librosa already in your deps)
    if sr != target_sr:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    # ensure float32 and shape (1, N)
    y = y.astype('float32')
    return y, sr

# Lazy-load SpeechBrain model for efficiency
_ECAPA_MODEL = None
def get_ecapa_model(device='cpu'):
    global _ECAPA_MODEL
    if _ECAPA_MODEL is None:
        try:
            from speechbrain.pretrained import EncoderClassifier
        except Exception as e:
            raise RuntimeError("speechbrain not installed or import failed: " + str(e))
        _ECAPA_MODEL = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )
    return _ECAPA_MODEL

def extract_ecapa_embedding(wav_path, device='cpu'):
    """
    Returns a 1-D numpy array (L2-normalized) speaker embedding using ECAPA-TDNN.
    """
    y, sr = _read_wav_for_speaker(wav_path, target_sr=16000)
    model = get_ecapa_model(device=device)
    # model expects a torch tensor [batch, time], float32
    import torch
    sig = torch.tensor(y).unsqueeze(0)  # shape (1, N)
    with torch.no_grad():
        emb = model.encode_batch(sig)  # returns torch.Tensor [1, emb_dim]
    emb = emb.squeeze(0).cpu().numpy()
    # L2-normalize
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb
