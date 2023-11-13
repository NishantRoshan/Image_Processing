import cv2
import numpy as np

# Usage
def solution(image_path):
    image= cv2.imread(image_path)
    ######################################################################
    ######################################################################
    '''
    The pixel values of output should be 0 and 255 and not 0 and 1
    '''
    #####  WRITE YOUR CODE BELOW THIS LINE ###############################
    image = np.float32(image)
    orig = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = image.copy().astype(np.uint8)
    # print(image.shape)

    Z = image.reshape(-1)
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    image = res.reshape((image.shape))
    ret, image = cv2.threshold(image,50,255,cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image = np.zeros(image.shape, np.uint8)

    gray_blurred = cv2.blur(gray, (3, 3))
    ret,thresh = cv2.threshold(gray_blurred,230,255,cv2.THRESH_BINARY)

    detected_circles = cv2.HoughCircles(thresh,
                    cv2.HOUGH_GRADIENT, 1, 100, param1 = 200,
                param2 = 15, minRadius = 20)
    if detected_circles is None:
        cv2.drawContours(image, contours, -1, 255,-1)

    image = np.broadcast_to(image[..., None],(image.shape[0],image.shape[1],3))


    ######################################################################
    return image

