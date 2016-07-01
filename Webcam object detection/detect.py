#Webcam object detection
#Michael Nyamande 2016
import cv2
import sys
import os
#choose calissifer to use
cpath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cpath)
video_capture = cv2.VideoCapture(0)
count = 4
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # detect faces in image

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #save the image to the results folder
        outfname = "results/faces{}.jpg" .format(count)
        cv2.imwrite(os.path.expanduser(outfname), frame)
        print("face")
        count += 1

    # Display the resulting frame
    cv2.imshow('Video', frame)
    #check for cancel key press (q)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
