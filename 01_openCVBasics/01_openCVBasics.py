import numpy as np
import cv2 as cv 
from matplotlib import pyplot as plt 

#-----------------------------------------------------------

img = cv.imread('OneDrive\Documents\pythonProjects\photo01.jpg')

cv.imshow('Photo01', img)

cv.waitkey(0) & 0xFF
## 64bit
cv.destroyAllWindows()

# #-----------------------------------------------------------

img = cv.imread('OneDrive\Documents\pythonProjects\photo01.jpg', 0)
cv.imshow('Photo01', img)
timeout = cv.waitKey(0) & 0xFF

if timeout == 27:
    cv.destroyAllWindows()
elif timeout == ord('s'):
    cv.imwrite('photo01_saved.jpg', img)
    cv.destroyAllWindows()


# # #-----------------------------------------------------------

capture = cv.VideoCapture('OneDrive\Documents\pythonProjects\\video01.mp4')

while True:
    isTrue, frame = capture.read()
    cv.imshow('Video01', frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.detroyAllWindows()

# #-----------------------------------------------------------

img = cv.imread('OneDrive\Documents\pythonProjects\photo01.jpg')
plt.imshow(img, cmap= 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])
plt.show()

# #-----------------------------------------------------------

capture = cv.VideoCapture(0)

while(True):
    ret, frame = capture.read()
    # ret = cap.set(4, 240)
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    

    cv.imshow('frame', gray)
    if cv.waitkey(1) & 0xFF == ord('q'):
        break
capture.release()
cv.destroyAllWindows()

# #-----------------------------------------------------------

# SAVE IT 

capture = cv.VideoCapture(0)

#define the codec and create VideoWriter Object

fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi', fourcc 20.0, (640,480))

while(capture.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = cv.flip(frame, 0)

        #write the flipped frame
        out.write(frame)

        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

#release everything if job is finished
capture.release()
out.release()
cv.destroyAllWindows()

#-----------------------------------------------------------

img = np.zeros((512,512,3), np.uint8)
cv.line(img, (0,0), (511,511), (255,0,0), 5)
cv.rectangle (img,(387,0), (510,128), (0,255,0),3)
cv.circle(img,(447,63), 63, (0,0,255), -1)
cv.ellipse(img, (256,256), (100,50), 0,0,180,255,-1)
pts = np.array([[10,5], [20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
cv.polylines(img, [pts], True, (0,255,255))
#cv.polylines(img, [pts], False, (0,255,255))

font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img, 'OpenCV', (10,500), font, 4, (255,255,255),2, cv.LINE_AA)
#orig: cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.CV_AA)

cv.imshow('test', img)
cv.waitKey(0)

cv.imshow('test', img)
cv.waitKey(0)
