import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# We are importing OS so that we can use the OS commands
# We are going to do that all the images of a folder automatically get recognized and we will also extract the names of the images

path = 'ImageAttendance' # Giving path of the Images Folder

images = [] # For grabbing the images of the folder
classNames = [] # Creating a List of saving names of the Images
myList = os.listdir(path) # We are going to grab the names of the images of the folder
# We are going to use the above names and we will import the images
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)

# Starting the Encoding Process
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

# Mark Attendance
def markAttendace(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        print(myDataList)

        nameList = []

        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {dtString}')


# markAttendace('a')

encodeListKnown = findEncodings(images)
print(len(encodeListKnown))
print('Encoding Complete')

# Finding the Matches between our encodings (Using WebCam)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25) # It is 1/4th of the size of original image. We shorten the size so that our process can be fast
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # There may be more than one face in the image so we will use face locations for this
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # Comparing the faces
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame): # It will grab one face location from facesCurFrame list and it will grab encoding of the respective image
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            markAttendace(name)


    cv2.imshow('WebCam', img)
    cv2.waitKey(1)


