import cv2
import numpy as np
import face_recognition
import os

#consts
imagesPath = 'Images'
ratio = 4

#containers
encodings = []
names = []

def init():
    images = get_all_images()
    for img in images:
        encoded = encode_image(img)
        encodings.append(encoded)
        names.append(os.path.basename(img).split('.')[0])

    print('Encoding completed!')


def get_all_images():
    file_list = []
    for root, dirs, files in os.walk(imagesPath):
        for file in files:
            # append the file name to the list
            file_list.append(os.path.join(root, file))

    return file_list


def encode_image(path):
    # load image
    img = face_recognition.load_image_file(path)

    # convert image to RGB and resize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # encode image
    encoding = face_recognition.face_encodings(img)
    if not encoding:
        print("Could not find face in image: ",  path)
        return

    encoded_img = face_recognition.face_encodings(img)[0]

    return encoded_img


def recognize_face(encoded_face):
    # get distances to face
    face_dis = face_recognition.face_distance(encodings, encoded_face)

    # get nearest neighbor
    match_index = np.argmin(face_dis)

    # find if faces match
    matches = face_recognition.compare_faces([encodings[match_index]], encoded_face)
    if matches[0]:
        return match_index
    return -1


init()

# define a video capture object
vid = cv2.VideoCapture(0)

while True:

    # Capture the video frame by frame
    ret, frame = vid.read()
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_processes = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_processes = cv2.resize(frame_processes, (0, 0), None, 1 / ratio, 1 / ratio)

    faces_locations_list = face_recognition.face_locations(frame_processes)
    faces_encoding_list = face_recognition.face_encodings(frame_processes)

    for encoded_face, face_locations in zip(faces_encoding_list, faces_locations_list):
        y1, x2, y2, x1 = face_locations
        y1, x2, y2, x1 = y1 * ratio, x2 * ratio, y2 * ratio, x1 * ratio
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

        match_index = recognize_face(encoded_face)
        if match_index != -1:
            cv2.rectangle(frame, (x1, y2-20), (x2, y2), (255, 0, 255), cv2.FILLED)
            cv2.putText(frame, names[match_index], (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, .5, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
