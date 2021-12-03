import cv2
import numpy as np
import face_recognition

# Load Images
img_elon = face_recognition.load_image_file('Images\\Elon_musk_1.jpg')
img_elon = cv2.cvtColor(img_elon, cv2.COLOR_BGR2RGB)

img_test = face_recognition.load_image_file('Images\\Bill_gates_1.jpg')
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

# dispaly detected faces
faceLoc = face_recognition.face_locations(img_elon)[0]
cv2.rectangle(img_elon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

testLoc = face_recognition.face_locations(img_test)[0]
cv2.rectangle(img_test, (testLoc[3], testLoc[0]), (testLoc[1], testLoc[2]), (255, 0, 255), 2)

# Encode images
encodeElon = face_recognition.face_encodings(img_elon)[0]
encodeTest = face_recognition.face_encodings(img_test)[0]

result = face_recognition.face_distance([encodeElon], encodeTest)
print(result)


cv2.imshow('Elon Musk', img_elon)
cv2.imshow('Test', img_test)
cv2.waitKey(0)
