# TP Traitement D'images

#### Ben Crulis & Alexandre Chanson

Note envoyer à donatello.conte@univ-tours.fr avant le 10/02/19

## Exercice 1

```python
# Mon script OpenCV : Video_processing
# importation des librairies, opencv et numpy
import numpy as np
import cv2
#
def frame_processing(imgc):
return imgc
# Charge la video depuis le disque, ou la webcam avec 0 comme parametre
cap = cv2.VideoCapture('data/jimmy_fallon.mp4')
#Tant que la video n'est pas fini ou que l'user apuis sur la touche q
while (not (cv2.waitKey(40) & 0xFF == ord('q'))):
    # On lit une frame de la video
    ret, frame = cap.read()
    # si la frame existe
    if ret:
        # On copie la frame puis on passe la copie en greyscale
        img = frame.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #On applique une transformation a l'image
        gray = frame_processing(gray)
        #On affiche dans deux fenetres separées la video avant/apres transformation
        cv2.imshow('MavideoAvant', frame)
        cv2.imshow('MavideoApres', gray)
    else:
        print('video ended')
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
```

## Exercice 2

### Question 1

```python
def blur(intensity):
    def intern(img):
        kernel = np.ones((intensity,intensity),np.float32)/(intensity**2)
        return cv2.filter2D(img,-1,kernel)
    return intern

processing_f = blur(20)
```

Pour la question 1 nous utilisons un noyau de convolution pour appliquer un flou basique à l'image.

Puis un flou gaussien (le parametre doit etre un nombre pair):

```python
def gaussian_blur(intensity):
    def intern(img):
        return cv2.GaussianBlur(img,(intensity,intensity),0)
    return intern

processing_f = gaussian_blur(101)
```

![flou_gaussien_de_jimmy](img/flou_gaussien_de_jimmy.png)

Ou peux aussi effectuer un flou median:

```python
def median_blur(intensity):
    def intern(img):
        return cv2.medianBlur(img,intensity)
    return intern

processing_f = median_blur(21)
```

![jimmy_chauve](img/jimmy_chauve.png)

### Question 2

On explore d'abbord l'algorithm de sobel qui detecte les bords soit en vertical soit en horizontal.

```python
def sobel(size,x,y):
    def intern(img):
        return cv2.Sobel(img,cv2.CV_64F,x,y,ksize=size)
    return intern

sobelx = sobel(3,1,0)
sobely = sobel(15,0,1)

processing_f = compose(sobely, greyscale)
```

![jimmy_sobely](img/jimmy_sobely.png)

On voit qu'il y a très peux de detection sur les rideaux en arrière plan ouisque ils ont des contours verticaux, il semble egalement que l'algorithme trouve des contours parfetement droit on supose que se sont des artefacts de l'algorithme de compression utilisé par youtube.

On veux ensuite utiliser un autre algorithme (canny), on choisie de le reproduire en utilisant les kernel de convolution et d'autre fonctions basiques d'opencv plutot que d'utiliser la fonction toute faite Canny.




