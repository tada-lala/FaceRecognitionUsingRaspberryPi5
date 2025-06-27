from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, flash
from face_utils import FaceRecognizer
import cv2
import numpy as np
import threading
import time
import os
from werkzeug.utils import secure_filename
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
face_recognizer = FaceRecognizer()

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('Data', exist_ok=True)

# Configure logging
log_handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
log_handler.setLevel(logging.INFO)
app.logger.addHandler(log_handler)

# Global variables for frame processing
current_frame = None
frame_lock = threading.Lock()
processing_results = []

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_frames():
    global current_frame, processing_results
    
    # Try different camera indices
    cap = None
    for i in [0, 1, 2]:
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            break
    
    if not cap or not cap.isOpened():
        app.logger.error("Could not open video capture")
        # Fallback to test image
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_img, "Camera not available", (50, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        while True:
            ret, buffer = cv2.imencode('.jpg', test_img)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
        return
    
    while True:
        success, frame = cap.read()
        if not success:
            app.logger.warning("Failed to grab frame")
            break
        
        with frame_lock:
            current_frame = frame.copy()
        
        for result in processing_results:
            top, right, bottom, left = result["location"]
            name = result["name"]
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        try:
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            app.logger.error(f"Error encoding frame: {e}")
            break

def process_frames():
    global current_frame, processing_results, frame_lock
    
    while True:
        try:
            if current_frame is not None:
                with frame_lock:
                    frame_to_process = current_frame.copy()
                
                results = face_recognizer.recognize_faces(frame_to_process)
                
                with frame_lock:
                    processing_results = results
        except Exception as e:
            app.logger.error(f"Error in process_frames: {e}")
        
        time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        files = request.files.getlist('file')
        name = request.form.get('name', '').strip()
        
        if not name:
            flash('Please enter a name')
            return redirect(request.url)
        
        if not files or all(file.filename == '' for file in files):
            flash('No selected files')
            return redirect(request.url)
        
        saved_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                try:
                    file.save(filepath)
                    saved_files.append(filepath)
                except Exception as e:
                    app.logger.error(f"Error saving file {filename}: {e}")
                    flash(f'Error saving {filename}')
        
        if saved_files:
            try:
                processed_files = face_recognizer.add_new_face(name, saved_files)
                flash(f'Successfully added {len(processed_files)} images for {name}')
            except Exception as e:
                app.logger.error(f"Error adding new face: {e}")
                flash(f'Error processing images: {str(e)}')
        
        return redirect(url_for('upload_file'))
    
    known_faces = face_recognizer.get_known_faces()
    return render_template('upload.html', known_faces=known_faces)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recognize', methods=['POST'])
def recognize():
    global processing_results
    return jsonify({"faces": processing_results})

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Server Error: {error}", exc_info=True)
    return "500 Error: Something went wrong", 500

@app.errorhandler(404)
def not_found_error(error):
    app.logger.error(f"Not Found Error: {error}")
    return "404 Error: Page not found", 404

if __name__ == '__main__':
    processing_thread = threading.Thread(target=process_frames)
    processing_thread.daemon = True
    processing_thread.start()
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)