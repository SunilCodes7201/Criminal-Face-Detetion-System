from base64 import encode
# this module use for finding face encodings 
from gettext import npgettext
# this module is use to gateing text
from importlib.resources import path
# this module is use for importing packages for lab 
import os
from tkinter import Y, Entry
from unicodedata import name
# this module is use for finding a code frome a name 
import cv2
# cv2 in a opencv module which is most importent module in this project 
import numpy as np
# numpy module is use to find a min amd max values between the face codings 
import face_recognition
# this face_recognition module is work on the basic of dlib which provide 128 differend code of faces 
from datetime import datetime
# this module is use for getting a date and time after recognition of criminal 
import csv    
# this module is use for csv files (csv means coma saprated value)    


path = 'images'
images = []
personName = []
myList = os.listdir(path)
print(myList)

for cu_img in myList:
    current_img = cv2.imread(f'{path}/{cu_img}')
    images.append(current_img)
    personName.append(os.path.splitext(cu_img)[0])
print(personName)


def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = faceEncodings(images)
print("***** All  Encodings Eomplete *****")
# HOG algoretham is use to find the encodings of face 


def Verification(name):
    with open('Verification1.csv', 'r+') as f:
        myDataList = f.readline()
        nameList = []
        for Line in myDataList:
            entry = Line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tstr = time_now.strftime('%H:%M:%S')
            dstr = time_now.strftime('%D/%M/%Y')
            f.writelines(f'{name}, {tstr}, {dstr}')


cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame, (0,0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)


    facesCurrentFrame = face_recognition.face_locations(faces)
    encodeCurrentFrame = face_recognition.face_encodings(faces,facesCurrentFrame)


    for encodeFace, faceLoc in zip(encodeCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)


        matchIndex = np.argmin(faceDis)


        if matches[matchIndex]:
            name = personName[matchIndex].upper()
            print(name)

            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(frame, (x1,y2-35), (x2,y2),(0,255,0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            Verification(name)




    cv2.imshow('Camara', frame)
    if cv2.waitKey(10) == 13:
        break

cap.release()
cv2.destroyAllWindows() 




       

            



        



       

        
     

      


