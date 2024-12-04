import cv2 as cv
import numpy as np
import rospy
import math
from collections import deque

trip = False
passed_crosswalk = False

def line_follow(self, img):

    global trip
    global passed_crosswalk

    image = self.bridge.imgmsg_to_cv2(img, desired_encoding="bgr8")

    detect_clueboard(image)

    if not passed_crosswalk and (detect_crosswalk(image) or trip):
        trip = True
        command = task_save_andy(self,image)
        if trip:
            return command
        else:
            # rospy.sleep(1)
            passed_crosswalk = True
            return command

    filtered = line_img_filter(image)

    x_target, y_target = get_target_coord(filtered)

    return compute_twist(x_target,y_target)

lower_white = 210
upper_white = 255

lower_green = np.array([60,130,20])
upper_green = np.array([80,150,40])

def line_img_filter(img):

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    mask = cv.inRange(blurred, lower_white, upper_white)
    #mask2 = cv.inRange(img, lower_green, upper_green)

    # combined = cv.bitwise_or(mask1,mask2)
    result = cv.bitwise_and(img, img, mask=mask)
    # result[combined>0] = [255,255,255]
    # cv.imshow('Original', img)
    # cv.imshow('Filtered', result)
    # cv.waitKey(1)
    # cv.destroyAllWindows()
    return result

def get_target_coord(img):
    
    col_list = []
    for col in range(img.shape[1]):
        if 255 not in img[400:700,col]:
            col_list.append(col)

    if not col_list:
        x_target = 800

    else:
        x_target = sum(col_list)/len(col_list)
    
    row_list = []
    for row in range(400, img.shape[0]):
        if 255 not in img[row,:]:
            row_list.append(row)

    if not row_list:
        y_target = 400

    else:
        y_target = sum(row_list)/len(row_list)

    return x_target,y_target  

Kp_linear = 0.002 #0.0015
Kp_angular = 1.75
max_linear = 3.0  
max_angular = 3.0 

def compute_twist(x_target, y_target):
    
    x_length = 400 - x_target
    y_length = 800 - y_target
    # print("X,Y",x_length,y_length)
    distance = math.sqrt(x_length**2 + y_length**2)
    angle = math.atan2(x_length, y_length)
    # print(distance,angle)
    x_vel = min(Kp_linear * distance, max_linear) 

    ang_vel = max(min(Kp_angular * angle, max_angular), -max_angular)
    # print(x_vel,ang_vel)
    return x_vel, ang_vel

history = deque(maxlen=3)

def acc_comms(x_vel, yaw): 

    history.append((x_vel,yaw))

    x_final = 0
    yaw_final = 0

    for comm in history:
        x_final = x_final + comm[0]
        yaw_final = yaw_final + comm[1]

    return x_final/len(history),yaw_final/len(history)

lower_red = np.array([0,0,200])
upper_red = np.array([20,20,255])

def detect_crosswalk(img):

    row = img[600:650]
    row = row.reshape(50, -1, 3)

    red_mask = cv.inRange(row, lower_red, upper_red)

    red_pixels = np.sum(red_mask>0)

    if (red_pixels / 800) > 0.15:
        return True

    return False

fgbg = cv.createBackgroundSubtractorMOG2()
last_10 = deque(maxlen=10)
floor_it = False
last_x = -2

def task_save_andy(self, img):

    global floor_it
    global trip
    global last_x

    command = center_road(img, lower_red, upper_red)

    if command != (0,0):
        return command

    last_10.append(find_andy(img,fgbg))


    if floor_it:
        if last_10[0][1] != -1:
            last_x = last_10[0][1]
        for i in range(3):
            if last_10[i][0] or last_x < 300:
                return command
        
        trip = False
        print("go")
        return (50.0,0)


    for elem in last_10:
        if elem[0] or len(last_10)!=10:
            return command
        
    floor_it = True
    return command

def center_road(img, lower, upper):

    angle = get_angle(img, lower, upper)
    
    if abs(angle) > 2:
        return 0, -0.1*angle
    
    else:
        return 0,0

def get_angle(img, lower, upper):

    mask =  mask = cv.inRange(img, lower, upper)

    result = cv.bitwise_and(img, img, mask=mask)

    edges = cv.Canny(result,50,150)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    angles = []

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if abs(x1 - x2) > 30:
                    angle = math.atan2(y2 - y1, x2 - x1) * (180 / np.pi)
                    angles.append(angle)

        for line in lines:
            for x1, y1, x2, y2 in line:
                if abs(x1-x2) > 30:
                    cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.imshow("Detected Lines", img)
    cv.waitKey(1)

    return np.mean(angles) if angles else 0

def find_andy(img,fgbg):

    fgmask = fgbg.apply(img)

    contours, _ = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv.contourArea(contour) > 1600: 
            x, y, w, h = cv.boundingRect(contour)
            if y > 250 and y < 400:
                cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv.imshow('Motion Detection', img)
                cv.waitKey(1)
                return True, x
        # cv.imshow('Motion Detection', img)
        # cv.waitKey(1)   
    return False, -1

def line_follow_leaves(self,img):

    image = self.bridge.imgmsg_to_cv2(img, desired_encoding="bgr8")

    filtered = line_img_filter_leaves(image)

    x_target, y_target = get_target_coord(filtered)

    comm = compute_twist(x_target,y_target)
    return comm

lower_leaf = 180
upper_leaf = 255

def line_img_filter_leaves(img):

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray[:400,:] = 0
    blurred = cv.GaussianBlur(gray, (15, 15), 0)
    
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    # cleaned = cv.morphologyEx(blurred, cv.MORPH_CLOSE, kernel)
    # cleaned = cv.morphologyEx(cleaned, cv.MORPH_OPEN, kernel)
    
    mask = cv.inRange(blurred, lower_leaf, upper_leaf)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(mask)
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 300:
            cv.drawContours(filtered_mask, [contour], -1, 255, -1)

    result = cv.bitwise_and(img, img, mask=filtered_mask)
    result[filtered_mask>0] = 255
    cv.imshow("Detected Line", result)
    cv.waitKey(1)
    return result
    

lower_blue1 = np.array([80,0,0])
upper_blue1 = np.array([120,20,20])

lower_blue2 = np.array([190,90,90])
upper_blue2 = np.array([210,110,110])

lower_blue3 = np.array([110,10,10])
upper_blue3 = np.array([130,30,30])

def detect_clueboard(img):

    blue = filter_blue(img, lower_blue1, upper_blue1)
    gray = cv.cvtColor(blue, cv.COLOR_BGR2GRAY)
    ret, binarized = cv.threshold(gray,10,255,cv.THRESH_BINARY)

    erosionKernel = np.ones((3,3))
    eroded = cv.erode(binarized, erosionKernel, iterations = 1)

    dilationKernel = np.ones((4,4))
    dilated = cv.dilate(eroded, dilationKernel, iterations = 3)
    
    inverted = cv.bitwise_not(dilated)
    contours, _ = cv.findContours(inverted, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    filtered_contours = contours[1:]
    # cv.drawContours(gray, contours, -1, 255, 3)
    
    for contour in filtered_contours:
        area = cv.contourArea(contour)
        if area > 1400:
            print(area)
            cv.drawContours(gray,[contour], -1, 255, 3)
            return True

    # print(seen)
    # cv.imshow("boARD",gray)
    # cv.waitKey(1)  
    return False

def filter_blue(img, lower, upper):
    blue = cv.inRange(img, lower, upper)
    return cv.bitwise_and(img, img, mask=blue)

def detect_stuck(img):
    return