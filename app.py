from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load Haar cascades for face, eyes, and smile detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Define funny comments based on confidence levels
def get_funny_comment(score):
    if score < 30:
        return "Feeling a bit down, huh? Maybe it's time for a snack!"
    elif score < 40:
        return "Meh... Did you forget to smile today?"
    elif score < 60:
        return "Not bad! But you could be a bit happier!"
    elif score < 80:
        return "Looking good! Who knew confidence could be this easy?"
    else:
        return "Wow! You're a confidence superstar!"

def confidence_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return 0  # Return a confidence score of 0 if no face is detected

    confidence_level = 0  # Start with 0

    for (x, y, w, h) in faces:
        face_region = gray[y:y+h, x:x+w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(face_region, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))
        eye_count = len(eyes)

        # Detect smiles
        smiles = smile_cascade.detectMultiScale(face_region, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25))
        smile_count = len(smiles)

        # Determine confidence score based on features detected
        if smile_count > 0:  # Smiling
            confidence_level = 80 + (smile_count * 3)  # Adjust multiplier for more subtle changes
            confidence_level = min(confidence_level, 100)  # Cap at 100
        elif eye_count > 0:  # Neutral or slightly happy
            confidence_level = 50 + (eye_count * 3)  # Base score of 50, add bonus for eyes
            confidence_level = min(confidence_level, 60)  # Cap at 60
        else:  # No significant expression detected
            confidence_level = 20  # Set to a lower score

    # Cap confidence level between 0 and 100
    confidence_level = max(0, min(confidence_level, 100))

    return confidence_level  # Return just the score

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Calculate confidence score
    score = confidence_score(opencv_image)
    comment = get_funny_comment(score)
    return jsonify({"score": score, "comment": comment})

if __name__ == '__main__':
    app.run(debug=True, port=3000)
