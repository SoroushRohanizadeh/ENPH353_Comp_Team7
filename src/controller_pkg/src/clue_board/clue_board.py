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

    for cnt in filtered_contours:
    # Check if the contour is valid (non-empty)
        if len(cnt) > 0:
            epsilon = 0.02 * cv2.arcLength(cnt, True)  # Adjust epsilon if necessary
            
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.polylines(img, [approx], isClosed = True, color=255, thickness = 3)

    return True, gray

def preProcess(img):
    '''
    reduce dimensionality of img
    '''
    return img