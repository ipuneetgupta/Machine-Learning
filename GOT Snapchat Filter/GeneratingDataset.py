# GeneratingDataset
import numpy as np 
import cv2

cap=cv2.VideoCapture(0)

Cascade_Nose=cv2.CascadeClassifier("./third-party/Nose18x15.xml")
Cascade_Eye=cv2.CascadeClassifier("./third-party/frontalEyes35x16.xml")

label=input("Enter the name of person u want to take snapshot of nose and ear")
N_Shots=int(input("Enter the number of shots u want to take::") or 10)

SnapShots_Eyes_Nose=[]

while N_Shots:
	ret,frame=cap.read()

	if not ret:
		continue

	Noses=Cascade_Nose.detectMultiScale(frame,1.3,5)
	Eyes=Cascade_Eye.detectMultiScale(frame,1.3,5)

	#find the largest nose and eye 
	Noses=sorted(Noses,key=lambda x:x[2]*x[3],reverse=True)
	Eyes=sorted(Eyes,key=lambda x:x[2]*x[3],reverse=True)


	for N  in Noses:
	    x,y,w,h=N
	    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
          

	for E in Eyes:
		x1,y1,w1,h1=E
		cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(0,0,255),2)

	cv2.imshow("NoseAndEye",frame)

	stop=cv2.waitKey(1)&0xFF
	if stop==ord('q'):
	     break

	print(len(Noses),len(Eyes))

    
	if not Noses or not Eyes:
		continue
     
	Noses=[Noses[0]]
	Eyes=[Eyes[0]]

	for Nose , Eye in zip(Noses,Eyes):
		#first resize and stor the nose
		x,y,w,h=Nose
		Cropped_Nose=frame[y:y+h,x:x+w]
		Cropped_Nose=cv2.resize(Cropped_Nose,(100,100))
		SnapShots_Eyes_Nose.append(Cropped_Nose)
        
		#Now resize the eyes

		x1,y1,w1,h1=Eye
		Cropped_Eye=frame[y1:y1+h,x1:x1+w]
		Cropped_Nose=cv2.resize(Cropped_Eye,(100,100))
		SnapShots_Eyes_Nose.append(Cropped_Eye)

		N_Shots-=1

	#cv2.imshow("Eyes And Nose Filter",frame)

	stop=cv2.waitKey(1)&0xFF
	if stop==ord('q'):
	     break

SnapShots_Eyes_Nose=np.array(SnapShots_Eyes_Nose)

print(SnapShots_Eyes_Nose.shape)
print("SnapShots_Eyes_Nose Are TaKen::",len(SnapShots_Eyes_Nose))

np.save("./InfoAboutNoseEye/"+label,SnapShots_Eyes_Nose)

cap.release()
cv2.destroyAllWindows()       


