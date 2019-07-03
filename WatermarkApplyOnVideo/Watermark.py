from RecordVideo import RecordVideo
import numpy
import cv2

cap=cv2.VideoCapture(0)

file_path="./VideoRe/puneet.MP4"
framePerSec=24
typeVideo=RecordVideo(cap,filepath=file_path,res="480p")
out=cv2.VideoWriter(file_path,typeVideo.video_type,framePerSec,typeVideo.dims)





while(True):

	ret,frame=cap.read()
	 
	if ret==False:
		continue

	out.write(frame)


	cv2.imshow("frame",frame)
	stopkey=cv2.waitKey(0)&0xFF
	if stopkey==ord('q'):
		break

cap.release()
out.release()
cv2.destroyAllWindows()