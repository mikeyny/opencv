#Face and eye detection using opencv
#Michael Nyamande 2016
import cv2
import sys
import os
#create the face cascades
face_cascade = cv2.CascadeClassifier('face.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

for infname in sys.argv[1:]:
    #get images from cmd arguments
    image_path = os.path.expanduser(infname)
    img = cv2.imread(image_path)
    #resize image
    r = 500.0 / img.shape[1]
    dim = (500, int(img.shape[0] * r))
    r_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(r_img, cv2.COLOR_BGR2GRAY)
    #detect faces in image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        #draw a recangle over the face
        cv2.rectangle(r_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = r_img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray ,1.2  )
        for (ex, ey, ew, eh) in eyes:
            #draw a rectangle over the eyes
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        outfname = "temp/%s.faces.jpg" % os.path.basename(infname)
        cv2.imwrite(os.path.expanduser(outfname), r_img)
        print("face")
