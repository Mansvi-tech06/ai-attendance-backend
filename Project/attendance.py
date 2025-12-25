import cv2
import face_recognition
import pickle
import pandas as pd
from datetime import datetime
import os

MIN_SECONDS = 5   
VIDEO_PATH = "classroom.mp4"

with open("encodings.pickle", "rb") as f:
    known_encodings, known_names = pickle.load(f)

video = cv2.VideoCapture(0)
if not video.isOpened():
    print("ERROR: Video source not found or cannot be opened")
    exit()

attendance = {}
presence_time = {}

print("Attendance system running. Press ESC to stop.")

while True:
    ret, frame = video.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    for encoding, location in zip(encodings, locations):
        matches = face_recognition.compare_faces(known_encodings, encoding)
        name = "Unknown"

        current_time = datetime.now()

        if True in matches:
            name = known_names[matches.index(True)]

            if name not in presence_time:
                presence_time[name] = current_time
            else:
                diff = (current_time - presence_time[name]).seconds
                if diff >= MIN_SECONDS and name not in attendance:
                    attendance[name] = current_time.strftime("%H:%M:%S")

        top, right, bottom, left = location
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, name, (left, top-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow("AI Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()

# -------- SAVE ATTENDANCE --------
print("Saving attendance...")

students = os.listdir("dataset")
records = []
today = datetime.now().strftime("%Y-%m-%d")

for student in students:
    if student in attendance:
        records.append([student, "Present", attendance[student], today])
    else:
        records.append([student, "Absent", "-", today])

df = pd.DataFrame(records, columns=["Name", "Status", "Time", "Date"])

csv_name = f"attendance_{today}.csv"
excel_name = f"attendance_{today}.xlsx"

df.to_csv(csv_name, index=False)
df.to_excel(excel_name, index=False)

print("Attendance saved successfully")
print(df)
