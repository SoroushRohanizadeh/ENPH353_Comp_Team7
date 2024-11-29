import cv2 as cv
import numpy as np
import rospy

def line_follow(self, img):
    return (0, 0) # x, yaw

def line_img_filter(self, img):
    image = self.bridge.imgmsg_to_cv2(img, desired_encoding="bgr8")
    # print(image.shape)
    lower_bound = np.array([240,240,240])
    upper_bound = np.array([255,255,255])

    mask = cv.inRange(image, lower_bound, upper_bound)

    result = cv.bitwise_and(image, image, mask=mask)

    x_target,y_target = get_target_coord(self,result)

    print(col_val)
    cv.imshow('Original', image)
    cv.imshow('Filtered', result)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return result

def get_target_coord(self, img):
    
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

