# fake_webrtcvad.py
class Vad:
    def __init__(self, mode=0):
        pass
    def is_speech(self, data, sample_rate):
        return True

# Allow import like "import webrtcvad"
import sys
sys.modules['webrtcvad'] = sys.modules[__name__]
