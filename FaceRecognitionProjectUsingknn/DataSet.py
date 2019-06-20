import numpy as np
import cv2

#Take the Camera control
cap=cv2.VideoCapture(0)

label=input("Enter the Name of Person for which u want to take Photo::")
num=(int)(input("Enter the number shots u want to take") or 10)

#Load the haar casade or trained xml file for identification of frontal face Or this is classifier for frontal face
Cascadeclassifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Here list store the faces
SnapShots=[]

while(num):
	#Now read frame from camera and  it return bool  value to from which we know frame read is succeefull or not
	Capture=cap.read() #return tuple (bool,frame(numpy array))
	ref , frame = Capture

	#print(ref)

	if not ref:
		continue

	#Now we need face from frame for that we use our classifier

	faces=Cascadeclassifier.detectMultiScale(frame,1.3,5)#frame,scaling factor,minneighbour #list of face (x,y,w,h)
	#type(faces)
	
	#sorted the face according to area

	faces=sorted(faces,key=lambda a: a[2]*a[3],reverse=True)

	if not faces:
		continue

	for face in faces[:1]:
		x,y,w,h = face #unpacking
		Croping_Face=frame[y:y+h,x:x+w] #view of frame
		Croping_Face=cv2.resize(Croping_Face,(100,100))#not inplace #Here it return copy of resize face

		#Now we need to add face in our snapshot for future use
		SnapShots.append(Croping_Face)
		num-=1 #we complete one shot

		#Now show our camera that we complete the shot
	cv2.imshow("feed",frame)

	#Now we need something to terminate our loop if no of taken is very large it may be infinite

	if cv2.waitKey(1) & 0xFF ==ord('q'): #ord return the ascii value in pytho
		break


	#Now we want to save our snapshot list of face in file but first we need to convert our list int numpy array

SnapShots=np.array(SnapShots)

	#for path ./ this represent present directory and followed by name of file and after / it mean do whatever do inside this file 

datapath="./face_dataset/"

np.save(datapath+label,SnapShots)
	
print("Snaps are Taken ::",len(SnapShots))

cap.release()

cv2.destroyAllWindows()



	


