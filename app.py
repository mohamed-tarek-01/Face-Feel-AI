from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from mtcnn import MTCNN
import io
import threading

app = Flask(__name__)

# Load facial emotion classification model
model = tf.keras.models.load_model('models/model.h5')
label_encoders = {
    'angry': 0, 'disgust': 1, 'fear': 2, 
    'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6
}
inv_label_map = {v: k for k, v in label_encoders.items()}

# Load MTCNN face detector
detector = MTCNN()

# Function to detect and process faces in images
def preprocess_image(image, target_size=(48, 48)):
    image = Image.open(image).convert('RGB')  # Convert image to RGB
    image = np.array(image)

    faces = detector.detect_faces(image)  # Detect faces using MTCNN
    if len(faces) == 0:
        return None, None, None

    # Select the median-sized face
    faces = sorted(faces, key=lambda x: x['box'][2] * x['box'][3])
    median_index = len(faces) // 2
    x, y, w, h = faces[median_index]['box']

    # Crop and preprocess the detected face
    face = image[y:y+h, x:x+w]
    face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    face = cv2.resize(face, target_size, interpolation=cv2.INTER_LINEAR)
    
    face = face / 255.0  # Normalize pixel values
    face = np.expand_dims(face, axis=-1)  # Expand dimensions for model input

    return np.expand_dims(face, axis=0), (x, y, w, h), image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/our_work')
def our_work():
    return render_template('our_work.html')

@app.route('/about_me')
def about_me():
    return render_template('about_me.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image, face_coords, original_image = preprocess_image(file)

    if image is None:
        return jsonify({'error': 'No face detected in the image'}), 400

    # Make prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    predicted_emotion = inv_label_map.get(predicted_class, "Unknown")

    # Draw bounding box and label on the image
    x, y, w, h = face_coords
    cv2.rectangle(original_image, (x, y), (x+w, y+h), (255, 0, 0), 3)
    cv2.putText(original_image, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Encode the processed image
    _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
    img_bytes = io.BytesIO(img_encoded.tobytes())

    return send_file(img_bytes, mimetype='image/jpeg')

# Webcam control
webcam_active = False
webcam_thread = None

def run_webcam():
    global webcam_active
    cap = cv2.VideoCapture(0)

    while webcam_active:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
        faces = detector.detect_faces(frame)  # Detect faces

        for face in faces:
            x, y, w, h = face['box']
            face_crop = gray_frame[y:y+h, x:x+w]  
            
            if face_crop.size == 0:
                continue

            face_crop = cv2.resize(face_crop, (48, 48))  # Resize to model input size
            face_crop = face_crop / 255.0  # Normalize pixel values
            face_crop = np.expand_dims(face_crop, axis=-1)  
            face_crop = np.expand_dims(face_crop, axis=0)  

            # Make prediction
            prediction = model.predict(face_crop)
            predicted_class = np.argmax(prediction)
            predicted_emotion = inv_label_map.get(predicted_class, "Unknown")

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Facial Emotion Recognition - Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    global webcam_active, webcam_thread
    if not webcam_active:
        webcam_active = True
        webcam_thread = threading.Thread(target=run_webcam, daemon=True)
        webcam_thread.start()
    return jsonify({'status': 'Webcam started'})

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    global webcam_active
    webcam_active = False
    return jsonify({'status': 'Webcam stopped'})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
