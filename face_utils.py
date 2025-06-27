import os
import cv2
import numpy as np
import face_recognition
import pickle
from datetime import datetime

class FaceRecognizer:
    def __init__(self, data_dir="Data"):
        self.data_dir = data_dir
        self.pickle_file = "face_data.pkl"
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_face_data()
    
    def load_face_data(self):
        if os.path.exists(self.pickle_file):
            with open(self.pickle_file, "rb") as f:
                self.known_face_encodings, self.known_face_names = pickle.load(f)
            print("Loaded face data from pickle file.")
        else:
            self.train_model()
    
    def preprocess_image(self, image_path, output_dir):
        """Process an image using your preprocessing code"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Unable to read image: {image_path}")
        
        # Your preprocessing code adapted as a method
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            raise ValueError("No face detected in image")
        
        x, y, width, height = faces[0]
        x, y = max(0, x), max(0, y)
        face_crop = img[y:y+height, x:x+width]
        
        TARGET_SIZE = (224, 224)
        h, w, _ = face_crop.shape
        scale = min(TARGET_SIZE[0] / w, TARGET_SIZE[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_face = cv2.resize(face_crop, (new_w, new_h))
        
        pad_w = (TARGET_SIZE[0] - new_w) // 2
        pad_h = (TARGET_SIZE[1] - new_h) // 2
        padded_face = cv2.copyMakeBorder(
            resized_face, pad_h, TARGET_SIZE[1] - new_h - pad_h, 
            pad_w, TARGET_SIZE[0] - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        
        normalized_face = padded_face.astype(np.float32) / 255.0
        output_image = (normalized_face * 255).astype(np.uint8)
        
        # Save the processed image
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"processed_{timestamp}.jpg")
        cv2.imwrite(output_path, output_image)
        
        return output_path
    
    def add_new_face(self, name, image_paths):
        """Add new face(s) to the recognition system"""
        person_dir = os.path.join(self.data_dir, name)
        
        # Process each image
        processed_paths = []
        for img_path in image_paths:
            try:
                processed_path = self.preprocess_image(img_path, person_dir)
                processed_paths.append(processed_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        # Retrain the model with new faces
        self.train_model()
        return processed_paths
    
    def train_model(self):
        """Train the model with all available faces"""
        self.known_face_encodings = []
        self.known_face_names = []
        
        for person_name in os.listdir(self.data_dir):
            person_path = os.path.join(self.data_dir, person_name)
            if os.path.isdir(person_path):
                for img_name in os.listdir(person_path):
                    img_path = os.path.join(person_path, img_name)
                    try:
                        image = face_recognition.load_image_file(img_path)
                        encodings = face_recognition.face_encodings(image)
                        if len(encodings) > 0:
                            self.known_face_encodings.append(encodings[0])
                            self.known_face_names.append(person_name)
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
        
        # Save to pickle file
        with open(self.pickle_file, "wb") as f:
            pickle.dump((self.known_face_encodings, self.known_face_names), f)
        print("Model trained and saved with", len(self.known_face_names), "known faces.")
    
    def recognize_faces(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            recognized_faces = []
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)
                name = "Unknown"
                
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                
                recognized_faces.append({
                    "name": name,
                    "location": (top, right, bottom, left)
                })
            
            return recognized_faces
        except Exception as e:
            print(f"Error in face recognition: {e}")
            return []
    
    def get_known_faces(self):
        """Get list of all known faces"""
        return list(set(self.known_face_names))