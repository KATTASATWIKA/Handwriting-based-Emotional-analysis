from flask import Flask, render_template, request, redirect, url_for, flash
import cv2
import numpy as np
import os
import pickle
from scipy.ndimage import gaussian_filter

# Load the trained model and label encoder
with open('handwriting_knn_model.pkl', 'rb') as file:
    knn_model = pickle.load(file)
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def extract_features(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found or unable to read image at {image_path}")

    # Thresholding to create a binary image
    _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # Extract contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize feature values
    slant, size, spacing, pressure, baseline, speed = 0, 0, 0, 0, 0, 0

    # Calculate average letter size
    sizes = [cv2.contourArea(c) for c in contours]
    size = np.mean(sizes) if sizes else 0

    # Calculate average slant
    angles = [cv2.fitEllipse(c)[2] for c in contours if len(c) >= 5]
    slant = np.mean(angles) if angles else 0

    # Calculate spacing
    bounding_boxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[0])
    spacings = [bounding_boxes[i+1][0] - (bounding_boxes[i][0] + bounding_boxes[i][2]) for i in range(len(bounding_boxes) - 1)]
    spacing = np.mean(spacings) if spacings else 0

    # Calculate pressure
    pressure = np.mean(img)

    # Calculate baseline consistency
    if contours:
        bottom_edges = [cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours]
        baseline = np.std(gaussian_filter(bottom_edges, sigma=1))
    else:
        baseline = 0

    # Calculate speed
    perimeters = [cv2.arcLength(c, True) for c in contours]
    speed = np.mean(perimeters) if perimeters else 0

    return slant, size, spacing, pressure, baseline, speed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        features = extract_features(file_path)
        features = np.array(features).reshape(1, -1)
        prediction = knn_model.predict(features)
        emotion = label_encoder.inverse_transform(prediction)[0]
        return render_template('result.html', emotion=emotion, file_path=file_path)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
