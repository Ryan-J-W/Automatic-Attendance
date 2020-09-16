import face_recognition
import cv2
import numpy as np
from gtts import gTTS
from playsound import playsound
import pgi
import gi
from playsound import playsound
welcomed_list = []

video_capture = cv2.VideoCapture(0)

ryan_image = face_recognition.load_image_file('ryan.jpg')
ryan_face_encoding = face_recognition.face_encodings(ryan_image)[0]

obama_image = face_recognition.load_image_file('obama.jpg')
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

adam_image = face_recognition.load_image_file('adam.jpg')
adam_face_encoding = face_recognition.face_encodings(adam_image)[0]


known_face_encodings = [ryan_face_encoding, adam_face_encoding]
known_face_names = ['Ryan Whyner', 'Adam Whyner', "Barrack Obama"]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True



def welcome_student(name, welcomed_list):
    if name in welcomed_list:
        return welcomed_list
    welcomed_list.append(name)
    text = "Welcome,,,, %s, you are all checked in!" % name
    language = 'en'
    if name == 'Unknown':
        text = 'Your face is not recognized, please register and come back soon.'

    myobj = gTTS(text=text, lang=language, slow=False)
    try:
        file = name.split()[0]+name.split()[1]
        file += '.wav'
        myobj.save(file)
        playsound(file)
    except IndexError:
        pass






    return welcomed_list




while True:

    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)

    name = 'Unknown'
    rgb_small_frame = small_frame[:,:,::-1]
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_name = []
        best_match_index = 0
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            if name not in welcomed_list:
                welcomed_list = welcome_student(name, welcomed_list)

    name = known_face_names[best_match_index]
    face_names = [name]



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






    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow('Video', frame)


video_capture.release()
cv2.destroyAllWindows()

