import cv2
import numpy as np

def detectClueBoard(img):
    img = preProcess(img)
    detected, transformed_img = detectBoard(img)
    if not detected:
        return False, 0, str()
    return True, parseBoard(transformed_img)

def detectClueBoard_Debug(img):
    '''
    used only during development
    '''
    # img = preProcess(img)
    detected, transformed_img = detectBoard(img)
    return transformed_img

def parseBoard(img):
    '''
    Parse board for message and number
    @returns board number, message
    '''
    return 0, str()

def detectBoard(img):
    '''
    Look for a clue board in img
    @return True if board detected
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binarized = cv2.threshold(gray, 15,255,cv2.THRESH_BINARY)

    erosionKernel = np.ones((3,3))
    eroded = cv2.erode(binarized, erosionKernel, iterations = 1)

    dilationKernel = np.ones((3,3))
    dilated = cv2.dilate(eroded, dilationKernel, iterations = 4)

    inverted = cv2.bitwise_not(dilated)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
    # cv2.drawContours(gray, filtered_contours, -1, 255, 3)

    ret_img = img
    for cnt in filtered_contours:
        epsilon = 0.1 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:
            # cv2.polylines(gray, [approx], isClosed = True, color=255, thickness = 3)
            contour_corners = sort_corners(np.float32(approx))
            

            w, h = gray.shape
            image_corners = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
            
            matrix = cv2.getPerspectiveTransform(contour_corners, image_corners)
            ret_img = cv2.warpPerspective(gray, matrix, (w, h))

    return ret_img != img, ret_img

def preProcess(img):
    '''
    reduce dimensionality of img
    '''
    return img

def sort_corners(corners):
    x0, y0 = corners[0][0]
    x1, y1 = corners[1][0]
    x2, y2 = corners[2][0]
    x3, y3 = corners[3][0]

    # Find the top-left, top-right, bottom-left, bottom-right by comparing x and y coordinates
    if x0 < x1 and y0 < y1:
        top_left = corners[0]
        top_right = corners[1]
    elif x0 > x1 and y0 < y1:
        top_left = corners[1]
        top_right = corners[0]
    elif x0 < x1 and y0 > y1:
        bottom_left = corners[0]
        bottom_right = corners[1]
    else:
        bottom_left = corners[1]
        bottom_right = corners[0]

    if x2 < x3 and y2 < y3:
        bottom_left = corners[2]
        bottom_right = corners[3]
    elif x2 > x3 and y2 < y3:
        bottom_left = corners[3]
        bottom_right = corners[2]
    elif x2 < x3 and y2 > y3:
        top_left = corners[2]
        top_right = corners[3]
    else:
        top_left = corners[3]
        top_right = corners[2]

    corners = [top_left, top_right, bottom_right, bottom_left]
    return np.float32(corners)
