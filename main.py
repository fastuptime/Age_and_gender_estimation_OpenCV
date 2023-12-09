import cv2
import numpy as np
from PIL import ImageColor

def classify_age_gender(frame, age_model, gender_model):
    blob = cv2.dnn.blobFromImage(frame, 1, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

    age_model.setInput(blob)
    age_preds = age_model.forward()
    age_index = np.argmax(age_preds[0])
    age_list = ['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']
    age = age_list[age_index]

    gender_model.setInput(blob)
    gender_preds = gender_model.forward()
    gender_confidence = gender_preds[0][0]
    gender = f"Erkek ({gender_confidence:.2%})" if gender_preds[0][0] > 0.5 else f"Kadin ({gender_confidence:.2%})"

    return age, gender

age_net = cv2.dnn.readNetFromCaffe("deploy_age.prototxt", "age_net.caffemodel")
gender_net = cv2.dnn.readNetFromCaffe("deploy_gender.prototxt", "gender_net.caffemodel")

cap = cv2.VideoCapture(0) # cap = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    face_frame = frame.copy()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) > 0:
        (x, y, w, h) = faces[0] 
        face_roi = frame[y:y + h, x:x + w]
        age, gender = classify_age_gender(face_roi, age_net, gender_net)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 189, 95), 2) 

        text = f"Yas: {age}\nCinsiyet: {gender}"
        for i, line in enumerate(text.split('\n')):
            cv2.putText(frame, line, (x, y - 30 - (i * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 189, 95), 2)

    cv2.imshow("Yas ve Cinsiyet Tahmini", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
