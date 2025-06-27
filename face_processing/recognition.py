import cv2
import numpy as np
import os
import face_recognition
import pickle

DATASET_DIR = "Data/"
PICKLE_FILE = "facenet_model.pkl"

# Load or create known faces dataset
if os.path.exists(PICKLE_FILE):
    print("Loading known faces from pickle file...")
    with open(PICKLE_FILE, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
else:
    print("Processing images to extract face encodings...")
    known_face_encodings = []
    known_face_names = []

    for person_name in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person_name)
        if os.path.isdir(person_path):
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                image = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(image)

                if len(encodings) > 0:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(person_name)

    # Save to pickle for future use
    with open(PICKLE_FILE, "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)
    print(f"Saved encodings to {PICKLE_FILE}")

# Start webcam for real-time recognition
def recognize_faces():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = "I do not know you yet!"

            if True in matches:
                match_index = matches.index(True)
                name = known_face_names[match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Face Recognition (FaceNet)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting Face Recognition using FaceNet & dlib...")
    recognize_faces()
