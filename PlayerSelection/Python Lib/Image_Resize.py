# Image_Resize

import cv2

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

if __name__=="__main__":
 print("hello")