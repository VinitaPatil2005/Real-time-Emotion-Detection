import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path

# ==========================================
# Load Model
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "emotion_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

# ==========================================
# Emotion Labels
# ==========================================

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# ==========================================
# Prediction Function
# ==========================================

def predict_emotion(face_image):

    # Resize image
    image = cv2.resize(face_image, (48, 48))

    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize
    image = image.astype("float32") / 255.0

    # Reshape
    image = image.reshape(1, 48, 48, 1)

    # Predict
    prediction = model.predict(image, verbose=0)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction)

    return emotion_labels[predicted_class], confidence