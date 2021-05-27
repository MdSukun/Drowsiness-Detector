################################ FINAL YEAR PROJECT #################################################

### Done By - MD SUKUN UL QALB, MD SADIQUIDDIN , NAVEEN KR SINGH, RICHA PANDEY, AADITYA KR PASWAN ###

###################################### GCELT, SALT LAKE, KOLKATA ####################################

# Importing the Modules
import cv2
import dlib
from scipy.spatial import distance
from pygame import mixer

#Initialising the alarm sound
mixer.init()
sound = mixer.Sound("alarm.mp3")

# Method to calculate the Eye Aspect ratio
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    eye_aspect_ratio = (A + B) / (2.0 * C)
    return eye_aspect_ratio

# variable to count the number of frames
c = 0
No_eye = 0
#capturing the camera
cap = cv2.VideoCapture(0)

#Declaring the face detector having 68 landmarks
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#Now capturing every live video frame by frame
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    f = 0
    for face in faces:

        f = 1
        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []  #Declaring the Left eye and Right eye list to store the coordinates
        rightEye = []

        for n in range(36, 42):
            x = face_landmarks.part(n).x      #returns the x coordinates
            y = face_landmarks.part(n).y      #returns the y coordinates
            leftEye.append((x, y),)
            next_point = n + 1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n + 1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        left_EAR = calculate_EAR(leftEye)
        right_EAR = calculate_EAR(rightEye)

        EAR = (left_EAR + right_EAR) / 2
        EAR = round(EAR, 2)
        cv2.putText(frame, "Final Year Project,IT (GCELT)", (20, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 4)

        # Code to calculate number of frames
        if EAR < 0.24:
            c = c + 1
        else:
            c = 0

        # If frames is below 5 then it is an eye blink
        if c > 10:
            cv2.putText(frame, "DROWSY", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            cv2.putText(frame, "Are you Sleepy?", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
            cv2.putText(frame, "Final Year Project,IT (GCELT)", (20, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 4)
            # Sounding the alarm if found drowsy
            try:
                sound.play()
            except:
                pass

            print("Drowsy")
            c = 0

        print(EAR)
    # Code to check if we are looking in other direction eg - sideways or below
    if f == 0:
        No_eye = No_eye + 1
    else:
        No_eye = 0

    if No_eye > 50:
        cv2.putText(frame, "DROWSY", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        cv2.putText(frame, "Are you Sleepy?", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
        cv2.putText(frame, "Final Year Project,IT (GCELT)", (20, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 4)
        try:
            sound.play()
        except:
            pass

    cv2.imshow("Drowsiness Detection Camera", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()

######################################## THANK YOU #############################################
