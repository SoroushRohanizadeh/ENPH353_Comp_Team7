import cv2
import numpy as np

def detectClueBoard(img):
    img = preProcess(img)
    if not boardDetected(img):
        return False, 0, str()
    return True, parseBoard(img)

def detectClueBoard_Debug(img):
    '''
    used only during development
    '''
    return preProcess(img)

def parseBoard(img):
    '''
    Parse board for message and number
    @returns board number, message
    '''
    return 0, str()

def boardDetected(img):
    '''
    Look for a clue board in img
    @return True if board detected
    '''
    return False

def preProcess(img):
    '''
    reduce dimensionality of img
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binarized = cv2.threshold(img, 15,255,cv2.THRESH_BINARY)
    return binarized