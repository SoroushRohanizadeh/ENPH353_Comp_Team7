import cv2 as cv
import numpy as np
import rospy
import math

def line_follow(self, img):

    image = self.bridge.imgmsg_to_cv2(img, desired_encoding="bgr8")

    filtered = line_img_filter(image)

    x_target, y_target = get_target_coord(filtered)

    return compute_twist(x_target,y_target)

def line_img_filter(img):

    lower_bound = np.array([240,240,240])
    upper_bound = np.array([255,255,255])

    mask = cv.inRange(img, lower_bound, upper_bound)

    result = cv.bitwise_and(img, img, mask=mask)

    # cv.imshow('Original', image)
    # cv.imshow('Filtered', result)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return result

def get_target_coord(img):
    
    col_list = []
    for col in range(img.shape[1]):
        if [255,255,255] not in img[400:,col]:
            col_list.append(col)

    if not col_list:
        x_target = 800

    else:
        x_target = sum(col_list)/len(col_list)
    
    row_list = []
    for row in range(400, img.shape[0]):
        if [255,255,255] not in img[row,:]:
            row_list.append(row)

    if not row_list:
        y_target = 400

    else:
        y_target = sum(row_list)/len(row_list)

    return x_target,y_target  

Kp_linear = 0.5
Kp_angular = 2.0
max_linear = 1.0  
max_angular = 4.0 

def compute_twist(x_target, y_target):
    
    x_length = 400 - x_target
    y_length = 800 - y_target
    # print("X,Y",x_length,y_length)
    distance = math.sqrt(x_length**2 + y_length**2)
    angle = math.atan2(x_length, y_length)
    # print(distance,angle)
    x_vel = min(Kp_linear * distance, max_linear) 

    ang_vel = max(min(Kp_angular * angle, max_angular), -max_angular)
    print(x_vel,ang_vel)
    return x_vel, ang_vel