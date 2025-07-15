import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ✅ Path to your model and cascade file (adjust only if needed)
MODEL_PATH = r'C:\Users\vanam\PycharmProjects\pythonProject\Proj2\emotion_model.h5'

CASCADE_PATH = r'C:\Users\vanam\PycharmProjects\pythonProject\Proj2\haarcascade_frontalface_default.xml'

# ✅ Load the trained model (.h5)
model = load_model(MODEL_PATH)

# ✅ Load Haar Cascade for face detection (.xml)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Emotion labels (should match your dataset folders if trained that way)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ✅ Start webcam
cap = cv2.VideoCapture(0)
print("[INFO] Press 'q' to exit webcam window.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = grayscale[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_gray, (48, 48))
        roi_normalized = roi_resized.astype('float32') / 255.0
        roi_input = np.expand_dims(roi_normalized.reshape(48, 48, 1), axis=0)

        prediction = model.predict(roi_input, verbose=0)
        emotion = emotion_labels[np.argmax(prediction)]

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Real-Time Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
