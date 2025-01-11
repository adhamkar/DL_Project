# emotion_detection.py
import cv2
from deepface import DeepFace
import time

class EmotionDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_emotion(self,duration=10):
        """
        Captures frames from the webcam for a specified duration and detects the dominant emotion.
        Returns the detected emotion or None if no face is detected.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        emotions = []
        start_time = time.time()
        while time.time() - start_time < duration:  # Capture frames for the specified duration
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                result = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                emotions.append(emotion)
            except Exception as e:
                print(f"DeepFace error: {e}")
            #emotion = None

        cap.release()

        if not emotions:
            return None

        from collections import Counter
        emotion_counter = Counter(emotions)
        dominant_emotion = emotion_counter.most_common(1)[0][0]
        return dominant_emotion