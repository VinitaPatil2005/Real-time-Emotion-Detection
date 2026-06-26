import cv2
from predict import predict_emotion

image = cv2.imread("../assets/test.jpg")

emotion, confidence = predict_emotion(image)

print("Emotion :", emotion)
print("Confidence :", confidence)