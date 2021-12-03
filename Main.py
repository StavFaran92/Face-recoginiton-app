import cv2
import numpy as np
import face_recognition
import os

imagesPath = 'Images'

def getAllImages():
    filelist = []
    for root, dirs, files in os.walk(imagesPath):
        for file in files:
            # append the file name to the list
            filelist.append(os.path.join(root, file))

    return filelist

def encodeImage(path):
    # load image
    img = face_recognition.load_image_file(path)

    # convert image to RGB and resize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (500, 500))

    #encode image
    encoding = face_recognition.face_encodings(img)
    if not encoding:
        print("Could not find face in image: ",  path)
        return

    encodedImg = face_recognition.face_encodings(img)[0]

    return encodedImg

encodings = []
names = []
images = getAllImages()
for img in images:
    encoded = encodeImage(img)
    encodings.append(encoded)
    names.append(os.path.basename(img).split('.')[0])

print('Encoding completed!')

def recognize_face(frame):
    faceDis = face_recognition.face_distance(encodings, frame)
    matchIndex = np.argmin(faceDis)
    matches = face_recognition.compare_faces(encodings, frame)
    if matches[matchIndex]:
        return matchIndex
    return -1

# define a video capture object
vid = cv2.VideoCapture(0)

while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    frameProcesses = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frameProcesses = cv2.resize(frameProcesses, (0, 0), None, 0.25, 0.25)

    # recognize_face(frame)
    faceLocations = face_recognition.face_locations(frameProcesses)
    faceEncoded = face_recognition.face_encodings(frameProcesses)

    for encodeFace, faceLoc in zip(faceEncoded, faceLocations):
        y1,x2,y2,x1 = faceLoc
        y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

        matchIndex = recognize_face(encodeFace)
        if matchIndex != -1:
            cv2.rectangle(frame, (x1, y2-35), (x2,y2), (255, 0, 255), cv2.FILLED)
            cv2.putText(frame, names[matchIndex], (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
