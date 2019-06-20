import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

#first extract all file in which snapshot store
filelist=os.listdir("./face_dataset")

labels=[f[:-4] for f in filelist  if  f.endswith(".npy")]

print(labels)

SnapShots=[]

#Extract all data which store in that file

for filename in filelist:
	tmp=np.load("./face_dataset/"+filename)
	SnapShots.append(tmp)
print(len(SnapShots))

SnapShots=np.concatenate(SnapShots,axis=0) #along column or row wise

SnapShots=SnapShots.reshape(SnapShots.shape[0],-1) #it make our data suitable for knn 
print(SnapShots.shape) #40:30000 in this 30000 is feature or ndim for cal nearest neighbour

labels=np.repeat(labels,10)

labels=labels.reshape(labels.shape[0],-1) # 40:1  #Here we make our label in row : col=1 for suitable to attach to our snapashot

dataset=np.hstack((SnapShots,labels))#along row or column wise

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(dataset[:,:-1],dataset[:,-1])

cap=cv2.VideoCaptur(0)

Cascadeclassifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:

	ref,frame=cap.read()

	if not ref :
		continue

	faces=Cascadeclassifier.detectMultiScale(frame,1.3,5)


	for face in faces:

		x,y,w,h=face

		Crpimage=frame[y:y+h,x:x+h]
		Crpimage=cv2.resize(Crpimage,(100,100))

		Crpimage=Crpimage.reshape(1,-1)#Here one image is 1 row with 30000 feature

		pred=knn.predict(Crpimage)
		print(pred)

		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

		cv2.putText(frame,pred[0],(x,y),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)

	cv2.imshow("feed",frame)

	if cv2.waitKey(1)&0xFF==ord('q'):
		break
cap.release()
cv2.detroyAllWindows()








