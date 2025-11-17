import cv2
import os

# Paths
dataset_path = "face_training/dataset"
os.makedirs(dataset_path, exist_ok=True)

# Create a Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Ask for Owner Name or ID
owner_name = input("Enter owner name or ID: ").strip()
person_path = os.path.join(dataset_path, owner_name)
os.makedirs(person_path, exist_ok=True)

# Initialize webcam
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow for Windows

count = 0
print("[INFO] Starting face capture. Look at the camera and wait...")

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to open webcam!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face = gray[y:y + h, x:x + w]
        file_path = os.path.join(person_path, f"{owner_name}_{count}.jpg")
        cv2.imwrite(file_path, face)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"Images Captured: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Capturing Faces", frame)

    # Stop if 'q' pressed or 50 images captured
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif count >= 50:
        break

print(f"[INFO] {count} face images saved to {person_path}")
cam.release()
cv2.destroyAllWindows()
