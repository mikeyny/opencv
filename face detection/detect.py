import numpy as np
import cv2
import sys
import os

face_cascade = cv2.CascadeClassifier('face.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')


for infname in sys.argv[1:]:
    image_path = os.path.expanduser(infname)
    img = cv2.imread(image_path)
    r = 500.0 / img.shape[1]
    dim = (500, int(img.shape[0] * r))
   # perform the actual resizing of the image and show it
    r_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(r_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(r_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = r_img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray ,1.2  )
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            print("eye")
        outfname = "temp/%s.faces.jpg" % os.path.basename(infname)
        cv2.imwrite(os.path.expanduser(outfname), r_img)
        print("face")
