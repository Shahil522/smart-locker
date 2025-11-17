import cv2
import os
import numpy as np

# Paths
dataset_path = "face_training/dataset"
trainer_path = "face_training/trainer"
os.makedirs(trainer_path, exist_ok=True)

# Initialize the recognizer and detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("[INFO] Training faces... Please wait.")

faces = []
ids = []
current_id = 0
label_map = {}

# Loop through dataset folder
for person_name in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_dir):
        continue

    label_map[current_id] = person_name
    for image_file in os.listdir(person_dir):
        path = os.path.join(person_dir, image_file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        faces.append(img)
        ids.append(current_id)
    current_id += 1

# Train the recognizer
if len(faces) == 0:
    print("[ERROR] No faces found. Please run capture_faces.py first.")
else:
    recognizer.train(faces, np.array(ids))
    recognizer.write(os.path.join(trainer_path, "trainer.yml"))
    np.save(os.path.join(trainer_path, "labels.npy"), label_map)
    print("[INFO] Training complete. Model saved to face_training/trainer/trainer.yml")
