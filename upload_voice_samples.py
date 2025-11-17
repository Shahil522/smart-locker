import requests

URL = "http://127.0.0.1:5000/api/auto_train"

files = [
    ("audio", open("static/audio/training_1763292093462.wav", "rb")),
    ("audio", open("static/audio/training_1763292109979.wav", "rb")),
    ("audio", open("static/audio/training_1763292114663.wav", "rb")),
    ("audio", open("static/audio/training_1763292120282.wav", "rb")),
    ("audio", open("static/audio/training_1763292124761.wav", "rb")),
]

response = requests.post(URL, files=files)
print(response.text)
