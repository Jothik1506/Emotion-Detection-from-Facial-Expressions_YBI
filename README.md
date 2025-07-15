# YBI_INTERNSHIP_PROJECT2

# Emotion Detection from Facial Expressions 😊

This project uses a Convolutional Neural Network (CNN) and OpenCV to detect human facial expressions in real-time via webcam and classify them into seven basic emotions:  
**Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral**

---

## 📁 Project Structure
Emotion-Detection-from-Facial-Expressions/
│
├── Proj2/
│ ├── app.py # Real-time emotion detector using webcam
│ ├── Main.py # Script for model training
│ ├── emotion_model.h5 # Trained Keras model (CNN)
│ ├── haarcascade_frontalface_default.xml # Haar Cascade for face detection
│ └── FER/ # Folder with train/test image data (not on GitHub due to size)
│ ├── train/
│ └── test/
│
├── README.md


---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Jothik1506/Emotion-Detection-from-Facial-Expressions.git
cd Emotion-Detection-from-Facial-Expressions

2. Create and Activate a Virtual Environment (Optional but Recommended)
python -m venv venv
venv\Scripts\activate   # On Windows

3. Install Dependencies
pip install -r requirements.txt

🏃 How to Run the Project
Option A: Train Your Own Model (Optional)
Run only if you want to retrain from scratch.

python Proj2/Main.py
#this will train the model and save it as emotion_model.h5


Option B: Run Real-Time Emotion Detection

python Proj2/app.py

Then:
Your webcam will open.
The model will predict the emotion in real time.
Press q to quit the webcam window.




---

### ✅ Next Steps

1. Copy this content into your `README.md` file.
2. Commit and push to GitHub.




requirements are
opencv-python
tensorflow
keras
numpy


