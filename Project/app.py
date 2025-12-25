from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import face_recognition
import pickle
from datetime import datetime
import numpy as np

app = Flask(__name__)
CORS(app)

# Load encodings
with open("encodings.pickle", "rb") as f:
    known_encodings, known_names = pickle.load(f)

attendance = {}

@app.route("/mark", methods=["POST"])
def mark_attendance():
    if "image" not in request.files:
        return jsonify({"error": "No image sent"}), 400

    file = request.files["image"]
    image_bytes = file.read()
    npimg = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    if len(encodings) == 0:
        return jsonify({"result": "No face detected"})

    encoding = encodings[0]
    matches = face_recognition.compare_faces(known_encodings, encoding)
    name = "Unknown"

    if True in matches:
        name = known_names[matches.index(True)]
        attendance[name] = datetime.now().strftime("%H:%M:%S")

    return jsonify({
        "name": name,
        "status": "Present" if name != "Unknown" else "Unknown"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
