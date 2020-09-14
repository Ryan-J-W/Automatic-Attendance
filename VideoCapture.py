import face_recognition
import cv2
import numpy as np
from gtts import gTTS
from pygame import mixer
import os
from playsound import playsound
welcomed_list = []

video_capture = cv2.VideoCapture(0)

ryan_image = face_recognition.load_image_file('ryan.jpg')
ryan_face_encoding = face_recognition.face_encodings(ryan_image)[0]

obama_image = face_recognition.load_image_file('obama.jpg')
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]


known_face_encodings = [ryan_face_encoding] #, biden_face_encoding]
known_face_names = ['1776', "Barrack Obama"]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True



def welcome_student(name, welcomed_list):
    if name in welcomed_list:
        return welcomed_list
    # The text that you want to convert to audio
    welcomed_list.append(name)
    text = "Welcome to class, %s" % name
    language = 'en'

    myobj = gTTS(text=text, lang=language, slow=False)

    myobj.save("%s.mp3"%name)

    # Playing the converted file

    return welcomed_list




while True:

    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:,:,::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_name = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

            welcomed_list = welcome_student(name, welcomed_list)


    process_this_frame =   not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *=4
        bottom *=4
        right *=4
        left *=4

        cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), cv2.INTER_AREA)

        cv2.rectangle(frame, (left, bottom- 35), (right, bottom), (0,0,255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left+6, bottom-6), font, 0.7, (255,255,255), 1)




    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()



