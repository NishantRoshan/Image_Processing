import cv2
import numpy as np

def get_kmeans(image):
    image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Z = image.reshape(-1)
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3

    _,labels,centers=cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    image = res.reshape((image.shape))

    image = cv2.threshold(image,50,255,cv2.THRESH_BINARY)[1]

    return image

def improve_mask(image, mask):
    for i,j in zip(*np.where(mask>0)):
        if image[i,j,2] - max(image[i,j,0],image[i,j,1]) < 55 :
            mask[i,j]=0

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS , (10,10))
    mask = cv2.dilate(mask, kernel)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel)
    return mask

def detect_sun(image):
    image = image.copy()
    image = image.astype(np.uint8)
    w,h,ch = image.shape
    # w = min([w,h])
    minR = int((w+h)/30)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray, (3, 3))
    ret,thresh = cv2.threshold(gray_blurred,230,255,cv2.THRESH_BINARY)
    
    detected_circles = cv2.HoughCircles(thresh,
                    cv2.HOUGH_GRADIENT, 1, 2000, param1 = 200,
                param2 = 15, minRadius = minR)
    if detected_circles is not None:
        # circles = np.round(circles[0, :]).astype("int")
        # # loop over the (x, y) coordinates and radius of the circles
        # for (x, y, r) in circles:
        #     # draw the circle in the output image, then draw a rectangle
        #     # corresponding to the center of the circle
        #     cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        return True
    return False

def solution(image_path):
    image = cv2.imread(image_path)
    ######################################################################
    ######################################################################
    '''
    The pixel values of output should be 0 and 255 and not 0 and 1
    '''
    #####  WRITE YOUR CODE BELOW THIS LINE ###############################
    image = np.float32(image)
    mask = get_kmeans(image)

    mask = improve_mask(image, mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    res = np.zeros(image.shape, np.uint8)
    # c = max(contours, key = cv2.contourArea)
    if not detect_sun(image):
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.001 * peri, True)
            cv2.drawContours(res, [approx], -1, [255,255,255],-1)

    # image = image[:,:,np.newaxis]
    # print(image.shape)
    ######################################################################
    return res

# x = solution('D:\\DATA\\Coding\\github\\EE604\\Assignment-2\\Q1\\test\\lava22.jpg')
# cv2.imwrite('n.png', x)
