import numpy as np
import cv2 as cv 
from matplotlib import pyplot as plt 

#-----------------------------------------------------------

img = cv.imread('Documents/pythonProjects/photo01.jpg')

cv.imshow('Photo01', img)

cv.waitKey(0) & 0xFF
## 64bit
cv.destroyAllWindows()

#-----------------------------------------------------------

img = cv.imread('Documents/pythonProjects/photo01.jpg', 0)
cv.imshow('Photo01', img)
timeout = cv.waitKey(0) & 0xFF

if timeout == 27:
    cv.destroyAllWindows()
elif timeout == ord('s'):
    cv.imwrite('photo01_saved.jpg', img)
    cv.destroyAllWindows()


#-----------------------------------------------------------

capture = cv.VideoCapture('Documents/pythonProjects/video01.mp4')

while True:
    isTrue, frame = capture.read()
    cv.imshow('Video01', frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.detroyAllWindows()

#-----------------------------------------------------------

img = cv.imread('Documents/pythonProjects/photo02.jpg')
plt.imshow(img, cmap= 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])
plt.show()

 