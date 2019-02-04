import numpy as np
import cv2
import functools


def compose2(f, g):
    return lambda *a, **kw: f(g(*a, **kw))

def compose(*fs):
    return functools.reduce(compose2, fs)

jimmy = 'jimmy_fallon.mp4'

cap = cv2.VideoCapture(jimmy)

def greyscale(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print(g.shape)
    return g

def blur(intensity):
    def intern(img):
        kernel = np.ones((intensity,intensity),np.float32)/(intensity**2)
        return cv2.filter2D(img,-1,kernel)
    return intern

def gaussian_blur(intensity):
    def intern(img):
        return cv2.GaussianBlur(img,(intensity,intensity),0)
    return intern

def median_blur(intensity):
    def intern(img):
        return cv2.medianBlur(img,intensity)
    return intern

def bilateral_blur(diameter, sigmaColor, sigmaSpace):
    def intern(img):
        return cv2.bilateralFilter(img,diameter,sigmaColor,sigmaSpace)
    return intern


def laplacian(img):
    return cv2.Laplacian(img,cv2.CV_64F)

def sobel(size,x,y):
    def intern(img):
        return cv2.Sobel(img,cv2.CV_64F,x,y,ksize=size)
    return intern

sobelx = sobel(3,1,0)
sobely = sobel(3,0,1)

def to_numpy(img):    
    return np.asfarray(img)*(1.0/255.0)
    
def from_numpy(mat):
    mat = mat*255
    return mat.astype(np.uint8, copy=True)
    

def Gx(intensity):
    kernel = np.matrix([[-1,0,1],[-1,0,1],[-1,0,1]])
    return cv2.filter2D(img,-1,kernel)

def Gy(intensity):
    kernel = np.matrix([[-1,-2,-1],[0,0,0],[-1,2,1]])
    return cv2.filter2D(img,-1,kernel)

def magn(img1, img2):
    img1 = to_numpy(img1)
    img2 = to_numpy(img2)
    return np.sqrt(np.square(img1)+np.square(img2))

def angle(gx, gy):
    return np.arctan(gy/gx)

processing_f = lambda img: (lambda greyblur: magn(Gx(greyblur), Gy(greyblur))) (gaussian_blur(5)(id(img)))


for i in range(50):
    cap.read()

while True:
    ret, frame = cap.read()
    if ret:
        img = greyscale(frame.copy())
        
        res = processing_f(img)
        
        cv2.imshow("before", frame)
        cv2.imshow("after", res)
    else:
        print('video ended')
        break
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


