import cv2

cap = cv2.VideoCapture(0)

face_classifier = cv2.CascadeClassifier("frontal_face.xml")
eye_classifier = cv2.CascadeClassifier("eye.xml")

while True:
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x + w , y + h),(0,255,0),thickness=2)
    
    roi_frame = frame[y:y+h,x:x+w]
    roi_gray = gray[y:y+h,x:x+w]  
    
    eyes = eye_classifier.detectMultiScale(roi_gray,1.3,4)
    for(ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_frame,(ex,ey),(ex + ew , ey + eh),(255,0,0),thickness=1)
    
    cv2.imshow("original",frame)
    
    
    
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()    