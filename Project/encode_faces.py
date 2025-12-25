import face_recognition
import os
import pickle

DATASET_DIR = "dataset"
known_encodings = []
known_names = []

print("Encoding faces...")

for person_name in os.listdir(DATASET_DIR):
    person_folder = os.path.join(DATASET_DIR, person_name)

    if not os.path.isdir(person_folder):
        continue

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        image = face_recognition.load_image_file(image_path)

        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(person_name)

with open("encodings.pickle", "wb") as f:
    pickle.dump((known_encodings, known_names), f)

print("Encoding completed successfully!")
