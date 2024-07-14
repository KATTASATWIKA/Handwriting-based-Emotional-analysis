import cv2
import numpy as np
import pandas as pd
import os
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Define a function to extract features from an image
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
    slant = 0
    size = 0
    spacing = 0
    pressure = 0
    baseline = 0
    speed = 0

    # Example: Calculate average letter size
    sizes = [cv2.contourArea(c) for c in contours]
    if sizes:
        size = np.mean(sizes)
    else:
        print("No contours found for size calculation")
        size = 0  # Assign a default value

    # Example: Calculate average slant (approximation)
    angles = []
    for contour in contours:
        if len(contour) >= 5:  # fitEllipse requires at least 5 points
            (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
            angles.append(angle)
    if angles:
        slant = np.mean(angles)
    else:
        print("No angles calculated")
        slant = 0  # Assign a default value

    # Example: Calculate spacing (approximation)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes.sort(key=lambda b: b[0])  # Sort by x coordinate
    spacings = [bounding_boxes[i+1][0] - (bounding_boxes[i][0] + bounding_boxes[i][2]) for i in range(len(bounding_boxes) - 1)]
    if spacings:
        spacing = np.mean(spacings)
    else:
        print("No spacings calculated")
        spacing = 0  # Assign a default value

    # Example: Calculate pressure (mean pixel intensity)
    pressure = np.mean(img)

    # Example: Calculate baseline consistency (approximation)
    # Smooth the bottom edge of the text lines
    if contours:
        bottom_edges = [cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours]
        baseline = np.std(gaussian_filter(bottom_edges, sigma=1))
    else:
        print("No contours found for baseline calculation")
        baseline = 0  # Assign a default value

    # Example: Calculate speed (approximation)
    # Use the contours to estimate the smoothness of strokes
    perimeters = [cv2.arcLength(c, True) for c in contours]
    if perimeters:
        speed = np.mean(perimeters)
    else:
        print("No perimeters calculated")
        speed = 0  # Assign a default value

    return slant, size, spacing, pressure, baseline, speed

# Directory containing handwriting samples
data_dir = "C:/Users/katta/Downloads/hds-20240616T145221Z-001/hds"  # Replace with your actual local path

# Emotion categories
emotions = ["happy", "sad", "angry", "fear", "anxiety"]

# Initialize a list to store the features and labels
data = []

# Loop through each emotion category
for emotion in emotions:
    emotion_dir = os.path.join(data_dir, emotion)
    for sample in os.listdir(emotion_dir):
        sample_path = os.path.join(emotion_dir, sample)
        if os.path.isfile(sample_path):  # Ensure the sample path is a file
            print("Processing:", sample_path)
            features = extract_features(sample_path)
            data.append((*features, emotion))

# Create a DataFrame with the extracted features
columns = ['slant', 'size', 'spacing', 'pressure', 'baseline', 'speed', 'emotion']
df = pd.DataFrame(data, columns=columns)

# Save the DataFrame to a CSV
csv_path = 'handwriting_features.csv'
df.to_csv(csv_path, index=False)

print("Feature extraction and CSV file creation completed successfully.")

# Load the DataFrame from the CSV
df = pd.read_csv(csv_path)

# Preprocess the data
# Encode the labels
label_encoder = LabelEncoder()
df['emotion'] = label_encoder.fit_transform(df['emotion'])

# Split the dataset into training and test sets
X = df.drop('emotion', axis=1)
y = df['emotion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize lists to store the results
K = []
training = []
test = []
scores = {}

# Loop through different values of k
for k in range(2, 21):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    training_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    K.append(k)
    training.append(training_score)
    test.append(test_score)
    scores[k] = [training_score, test_score]

# Print the scores for each value of k
for keys, values in scores.items():
    print(keys, ':', values)


# Save the model with the best k to a file using pickle
best_k = max(scores, key=lambda k: scores[k][1])
best_knn_model = KNeighborsClassifier(n_neighbors=best_k)
best_knn_model.fit(X_train, y_train)

model_file_path = 'handwriting_knn_model.pkl'
with open(model_file_path, 'wb') as file:
    pickle.dump(best_knn_model, file)

print(f"K-Nearest Neighbors model with k={best_k} saved successfully.")
label_encoder_file_path = 'label_encoder.pkl'
with open(label_encoder_file_path, 'wb') as file:
    pickle.dump(label_encoder, file)

# Save the model with the best k to a file using pickle
model_file_path = 'handwriting_knn_model.pkl'
with open(model_file_path, 'wb') as file:
    pickle.dump(best_knn_model, file)

print(f"Label encoder and K-Nearest Neighbors model with k={best_k} saved successfully.")