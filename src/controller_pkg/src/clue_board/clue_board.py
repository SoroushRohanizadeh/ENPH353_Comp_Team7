import cv2
import numpy as np

BORDER_THRESHOLD = 15
LETTER_THRESHOLD = 70

def detectClueBoard(img):
    detected, transformed_img = detectBoard(img)
    if not detected:
        return False, 0, str()
    return True, parseBoard(transformed_img)

def detectClueBoard_Debug(img):
    '''
    used only during development
    '''
    detected, transformed_img = detectBoard(img)
    if not detected: return img
    letter_img = highlightLetters(transformed_img)
    return letter_img

def parseBoard(img):
    '''
    Parse board for message and number
    @returns board number, message
    '''
    return 0, str()

def highlightLetters(img):
    '''
    Identify and highlight the letters in the image
    '''
    ret, binarized = cv2.threshold(img, LETTER_THRESHOLD,255,cv2.THRESH_BINARY)

    dilationKernel = np.ones((2,2))
    dilated = cv2.dilate(binarized, dilationKernel, iterations = 1)

    inverted = cv2.bitwise_not(dilated)
    contours, _ = cv2.findContours(inverted, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    contours = contours[2:]

    # cv2.drawContours(img, contours, -1, 255, 3)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)

    return img

def detectBoard(img):
    '''
    given a raw image, determine if a clue board can be found. If found, 
        perform a perspective transform to center the image on the clueboard
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binarized = cv2.threshold(gray, BORDER_THRESHOLD,255,cv2.THRESH_BINARY)

    erosionKernel = np.ones((3,3))
    eroded = cv2.erode(binarized, erosionKernel, iterations = 1)

    dilationKernel = np.ones((3,3))
    dilated = cv2.dilate(eroded, dilationKernel, iterations = 4)

    inverted = cv2.bitwise_not(dilated)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
    # cv2.drawContours(gray, filtered_contours, -1, 255, 3)

    ret_img = img
    for cnt in filtered_contours:
        epsilon = 0.1 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:
            # cv2.polylines(gray, [approx], isClosed = True, color=255, thickness = 3)
            # contour_corners = np.float32(approx)
            contour_corners = sort_corners(approx)


            w, h = gray.shape
            image_corners = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
            
            matrix = cv2.getPerspectiveTransform(contour_corners, image_corners)
            ret_img = cv2.warpPerspective(gray, matrix, (w, h))

    return ret_img != img, ret_img

def sort_corners(corners):
    corners = [point[0] for point in corners]
    sorted_corners = sorted(corners, key=lambda x: x[0])
    
    left = sorted_corners[:2]
    right = sorted_corners[2:]

    left = sorted(left, key=lambda x: x[1])
    right = sorted(right, key=lambda x: x[1])
    
    return np.array([left[0], right[0], right[1], left[1]], dtype=np.float32)

