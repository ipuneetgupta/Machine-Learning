# NoseAndEyesRecogniation
import cv2
# Image_Resize
def image_resize(image,width=None,height=None,inter=cv2.INTER_AREA):
    dim=None
    ht,wt=image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:

        ratio=height/float(ht)
        dim=(int(ratio*wt),height)

    else:
        ratio=width/float(wt)
        dim=(width,int(ratio*ht))

    resized_image=cv2.resize(image,dim,interpolation=inter)

    return resized_image

cap=cv2.VideoCapture(0)
cascade_Nose=cv2.CascadeClassifier("./third-party/Nose18x15.xml")
cascade_Eye=cv2.CascadeClassifier("./third-party/frontalEyes35x16.xml")
cascade_face=cv2.CascadeClassifier("./third-party/haarcascade_frontalface_default.xml")
img_mous=cv2.imread("mustache.png",-1)
img_gla=cv2.imread("glasses.png",-1)
while True:
    ret,frame=cap.read()
    if ret is False:
        continue

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=cascade_face.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2BGRA)
    for face in faces:
        x,y,w,h=face

        face_portion_gray=gray[y:y+h,x:x+w]
        face_portion_color=frame[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        Noses=cascade_Nose.detectMultiScale(face_portion_gray,1.3,5)
        for nose in Noses:
            nx,ny,nw,nh=nose

            mustache_resize=image_resize(img_mous.copy(),width=nw)
            # print(img_mous.shape)
            mw,mh,mc=mustache_resize.shape  # width,height,channel
            #print(mw,mh)
            for i in range(mw):
                for j in range(mh):
                    if mustache_resize[i,j][3]!=0: #if it is not complete transparent
                               face_portion_color[ny+i+int(nh/2.0),nx+j]=mustache_resize[i,j]


        Eyes=cascade_Eye.detectMultiScale(face_portion_gray,1.3,5)
        print(len(Eyes))
        for eye in Eyes:
            ex,ey,ew,eh=eye

            glasses_resize=image_resize(img_gla.copy(),width=ew)

            gw,gh,gc=glasses_resize.shape
            print(gw,gh)
            for i in range(gw):
                for j in range(gh):
                    if glasses_resize[i,j][3]!=0:
                        face_portion_color[ey+i,ex+j]=glasses_resize[i,j]


    frame=cv2.cvtColor(frame,cv2.COLOR_BGRA2BGR)    
    cv2.imshow("NoseAndEye",frame)

    if(cv2.waitKey(1)&0xFF==ord('q')):
         break


cap.release()
cv2.destroyAllWindows()
