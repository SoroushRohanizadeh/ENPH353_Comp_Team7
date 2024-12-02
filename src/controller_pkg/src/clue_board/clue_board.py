import cv2
import numpy as np

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

