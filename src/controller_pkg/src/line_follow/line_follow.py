import cv2 as cv
import numpy as np
import rospy
import math
from collections import deque

trip = False

def line_follow(self, img):

    global trip

    image = self.bridge.imgmsg_to_cv2(img, desired_encoding="bgr8")
    
    if detect_crosswalk(image) or trip:
        trip = True
        return task_save_andy(self,image)

    filtered = line_img_filter(image)

    x_target, y_target = get_target_coord(filtered)

    return compute_twist(x_target,y_target)

lower_bound = np.array([240,240,240])
upper_bound = np.array([255,255,255])

def line_img_filter(img):

    mask = cv.inRange(img, lower_bound, upper_bound)

    result = cv.bitwise_and(img, img, mask=mask)

    # cv.imshow('Original', img)
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

Kp_linear = 0.003
Kp_angular = 1.0
max_linear = 5.0  
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
    # print(x_vel,ang_vel)
    return x_vel, ang_vel

history = deque(maxlen=5)

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

    if (red_pixels / 800) > 0.3:
        return True

    return False

def task_save_andy(self, img):

    return center_road(self,img)

    #detect when andy crosses

    #wait

    #floor it

def center_road(self, img):

    angle = get_angle(img)
    
    if abs(angle) > 1:
        return 0, -0.1*angle
    
    else:
        return 0,0


def get_angle(img):

    mask =  mask = cv.inRange(img, lower_red, upper_red)

    result = cv.bitwise_and(img, img, mask=mask)

    edges = cv.Canny(result,50,150)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    angles = []

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                angle = math.atan2(y2 - y1, x2 - x1) * (180 / np.pi)
                angles.append(angle)

    # for line in lines:
    #     for x1, y1, x2, y2 in line:
    #         cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv.imshow("Detected Lines", img)
    # cv.waitKey(1)

    avg_angle = np.mean(angles) if angles else 0

    perp_angle = avg_angle
    # perp_angle = (perp_angle + 180) % 360 - 180
    print(perp_angle)
    return perp_angle