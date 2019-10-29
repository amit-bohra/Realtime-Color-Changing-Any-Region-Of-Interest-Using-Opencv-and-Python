import cv2
import numpy as np
from copy import deepcopy as dp

a=0
a=int(input('Enter 1 for Video Cam else 0 '))

if a==1:
    cap=cv2.VideoCapture(0)
    if cap.isOpened():
        ret,frame=cap.read()
        frame=cv2.flip(frame,1)
    else:
        ret=False
else:
    frame=cv2.imread('color_circle.jpg')
    frame=cv2.resize(frame,(512,512))

def func(x):
    pass

cv2.namedWindow('image',cv2.WINDOW_AUTOSIZE)
lh,ls,lv,hh,hs,hv=100,50,50,40,205,205


cv2.createTrackbar('LH','image',0,179,func)
cv2.createTrackbar('LS','image',70,255,func)
cv2.createTrackbar('LV','image',91,255,func)
cv2.createTrackbar('HH','image',41,179,func)
cv2.createTrackbar('HS','image',111,255,func)
cv2.createTrackbar('HV','image',186,255,func)

cv2.createTrackbar('H','image',32,179,func)
cv2.createTrackbar('S','image',80,255,func)
cv2.createTrackbar('V','image',72,255,func)
cv2.createTrackbar('k','image',7,255,func)

while True:
    if a==1:
        ret,frame=cap.read()
        frame=cv2.flip(frame,1)
    else:
        frame=cv2.imread('color_circle.jpg')
        frame=cv2.resize(frame,(512,512))
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    trame=dp(frame)

    lh=cv2.getTrackbarPos('LH','image')
    ls=cv2.getTrackbarPos('LS','image')
    lv=cv2.getTrackbarPos('LV','image')
    hh=cv2.getTrackbarPos('HH','image')
    hs=cv2.getTrackbarPos('HS','image')
    hv=cv2.getTrackbarPos('HV','image')
    
    h=cv2.getTrackbarPos('H','image')
    s=cv2.getTrackbarPos('S','image')
    v=cv2.getTrackbarPos('V','image')

    k=cv2.getTrackbarPos('k','image')
    if k<1:
        k=1
    if k%2==0:
        k+=1
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(k,k))
    lower=np.array([lh,ls,lv])
    higher=np.array([lh+hh,ls+hs,lv+hv])

    mask=cv2.inRange(frame,lower,higher)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    inv_mask=cv2.bitwise_not(mask)
    index=mask.nonzero()
    trame[index]=[h,s,v]
    color=cv2.bitwise_and(frame,frame,mask=mask)
    remaining=cv2.bitwise_and(frame,frame,mask=inv_mask)
    img3=np.hstack((color,remaining))
    img4=np.hstack((frame,trame))
    cv2.imshow('mask',mask)
    cv2.imshow('img3',img3)
    cv2.imshow('img4',img4)
    if cv2.waitKey(1)==27:
        break
    if cv2.waitKey(1)==ord('p'):
        print('lower : ({},{},{})'.format(lh,ls,lv))
        print('higher : ({},{},{})'.format(lh+hh,ls+hs,lv+hv))
cv2.destroyAllWindows()
if a==1:
    cap.release()
    
