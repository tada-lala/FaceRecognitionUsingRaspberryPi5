import cv2
import os
import numpy as np

# Set input and output directories Nive,teena,
INPUT_DIR =r"Data/Ms. Chris Jenifer"  # Change this to your folder containing images
OUTPUT_DIR = r"Data/Ms. Chris Jenifer"  # Change this to your desired output folder

TARGET_SIZE = (224, 224)  # Define the target size for resizing

def preprocess_images():
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # Load Haar cascade
    
    for filename in os.listdir(INPUT_DIR):
        img_path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping {filename}: Unable to read image.")
            continue
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for face detection
        faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))  # Detect faces
        
        if len(faces) == 0:
            print(f"No face detected in {filename}, skipping...")
            continue
        
        # Assume the first detected face is the primary face
        x, y, width, height = faces[0]
        x, y = max(0, x), max(0, y)  # Ensure coordinates are non-negative
        face_crop = img[y:y+height, x:x+width]  # Crop the face
        
        # Resize while maintaining aspect ratio
        h, w, _ = face_crop.shape
        scale = min(TARGET_SIZE[0] / w, TARGET_SIZE[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_face = cv2.resize(face_crop, (new_w, new_h))
        
        # Add padding to match TARGET_SIZE
        pad_w = (TARGET_SIZE[0] - new_w) // 2
        pad_h = (TARGET_SIZE[1] - new_h) // 2
        padded_face = cv2.copyMakeBorder(
            resized_face, pad_h, TARGET_SIZE[1] - new_h - pad_h, pad_w, TARGET_SIZE[0] - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)  # Black padding
        )
        
        # Normalize pixel values to range [0,1]
        normalized_face = padded_face.astype(np.float32) / 255.0
        
        # Convert back to uint8 for saving
        output_image = (normalized_face * 255).astype(np.uint8)
        
        # Save preprocessed image
        output_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(output_path, output_image)
        print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    preprocess_images()
