import cv2
import numpy as np

def dist(p1,p2):
    return max(abs(p1[0]-p2[0]),abs(p1[1]-p2[1]))

def solution(audio_path):
    ############################
    ############################

    class_name = 'real'

    img = cv2.imread(audio_path)
    img = cv2.copyMakeBorder(img,20,20,20,20,cv2.BORDER_CONSTANT,value=[255,255,255])
    img = cv2.bilateralFilter(img, 15, 75, 75)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = 255 - gray
    gray = cv2.threshold(gray, 25, 255, cv2.THRESH_TOZERO )[1]
    corners = cv2.goodFeaturesToTrack(gray,500,0.01,5)
    corners = np.intp(corners).squeeze()
    # print(corners)
    top_points = [(x,y) for x,y in corners if gray[y,x-5]==0 and gray[y,x+5]==0 and gray[y-5,x]==0]

    bot_pnts = [(x,y) for x,y in corners if gray[y+5,x]==0]
    m=max(corners, key=lambda x:x[1])[1]

    l = [x for x,y in corners if m-y<=5]
    mid = int((min(l)+max(l))/2)
    left_n = sum(1 for x,y in top_points if mid-x>10)
    right_n = sum(1 for x,y in top_points if x-mid>10)
    if left_n!=4 or right_n!=5:
        class_name = 'fake'
    l=[(mid,int((min(corners, key=lambda x:x[1])[1]+m)/2))]
    
    p=l[0]
    d = (mid - max([x for x,y in top_points if mid-x>10]))/2
    cnt = sum(1 for x,y in corners if dist(p,(x,y))<d)
    if cnt < 5:
        class_name = 'fake'


    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    # class_name = 'fake'
    return class_name

