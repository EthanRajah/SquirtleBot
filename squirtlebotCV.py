import cv2 as cv
import numpy as np

# Create pop up window and access webcam
cv.startWindowThread()
cap = cv.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 420)

# import cascade file for facial recognition -- pre-trained HAAR model for face recognition from OpenCV
# Model is loaded to be used as a prediction classifier on webcam data
faceCascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    success, img = cap.read()
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Getting corners around the face
    # 1.3 = scale factor, 5 = minimum neighbor
    faces = faceCascade.detectMultiScale(imgGray, 1.3, 5) 
    # drawing bounding box around face
    for (x, y, w, h) in faces:
        img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv.imshow('face_detect', img)
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyWindow('face_detect')
