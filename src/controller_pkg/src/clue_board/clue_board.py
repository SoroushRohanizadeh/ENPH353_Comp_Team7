import cv2
import numpy as np
from collections import Counter
from skimage.metrics import structural_similarity as ssim

BORDER_COLOR_THRESHOLD = 15

TRANSFORMED_CB_WIDTH = 800
TRANSFORMED_CB_HEIGHT = 800

LETTER_COLOR_THRESHOLD = 70
LETTER_HEIGHT_TOLERANCE = 5
MIN_LETTER_HEIGHT = 50
LETTER_BORDER_THICKNESS = 3

FIZZ_ICON_PATH = "clue_board/reference_images/fizz_detective.png"
LETTER_IMG_PATH = "clue_board/reference_images/"
NUM_MATCHES_FOR_HOMOGRAPHY = 8



class ClueBoard:

    def __init__(self):
        return

    def detectClueBoard(self, img):
        detected, transformed_img = self.detectBoard(img)
        if not detected: return False, 0, str()
        return True, self.parseBoard(transformed_img)

    def detectClueBoard_Debug(self, img):
        '''
        used only during development
        '''
        detected, transformed_img = self.detectBoard(img)
        if not detected: return img

        # top, bottom = self.highlightLetters(transformed_img)
        top = self.parseBoard(transformed_img)
        return top
    
    def parseBoard(self, img):
        '''
        Parse board for message and number
        @returns board number, message
        '''
        top, bottom = self.highlightLetters(img)

        top_chars = []
        for letter in top:
            top_chars.append(self.imgToChar(letter))

        bottom_chars = []
        for letter in bottom:
            bottom_chars.append(self.imgToChar(letter))

        # reference = cv2.imread(LETTER_IMG_PATH + "I.png", cv2.IMREAD_GRAYSCALE)
        # _, reference = cv2.threshold(reference, LETTER_COLOR_THRESHOLD,255,cv2.THRESH_BINARY)
        # print(self.imgDifference(top[0], reference))
        return bottom[1]

    def imgDifference(self, img1, img2):
        height = max(img1.shape[0], img2.shape[0])
        width = max(img1.shape[1], img2.shape[1])

        img1 = self.padImg(img1, height, width)
        img2 = self.padImg(img2, height, width)

        # return np.mean((img1 - img2) ** 2)
        return ssim(img1, img2, full = True)[0]

    def padImg(self, img, height, width):
        top = (height - img.shape[0]) // 2
        bottom = height - img.shape[0] - top
        left = (width - img.shape[1]) // 2
        right = width - img.shape[1] - left 
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, 255)

    def imgToChar(self, letter):
        '''
        Return the char that is most likely present in the letter image
        '''
        return ''

    def highlightLetters(self, img):
        '''
        Identify and highlight the letters in the image
        '''
        ret, binarized = cv2.threshold(img, LETTER_COLOR_THRESHOLD,255,cv2.THRESH_BINARY)

        dilationKernel = np.ones((2,2))
        dilated = cv2.dilate(binarized, dilationKernel, iterations = 1)

        inverted = cv2.bitwise_not(dilated)

        cv2.fillPoly(inverted, [np.int32(self.icon_corners)], color = 0)

        contours, _ = cv2.findContours(inverted, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        contours = sorted(contours, key = cv2.contourArea, reverse = True)
        contours = contours[2:]
        # cv2.drawContours(img, contours, -1, 255, 3)

        heights = []
        for contour in contours:
            h = cv2.boundingRect(contour)[3]
            if (h > MIN_LETTER_HEIGHT):
                heights.append(h)
        contours = sorted(contours, key = lambda contour: cv2.boundingRect(contour)[0]) #sort by x location

        counter = Counter(heights)
        letter_height, _ = counter.most_common(1)[0] # extract most common height value 

        top_letters = []
        bottom_letters = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (np.abs(h - letter_height) < LETTER_HEIGHT_TOLERANCE):
                letter = dilated[y - LETTER_BORDER_THICKNESS 
                            : y + h + LETTER_BORDER_THICKNESS, 
                            x - LETTER_BORDER_THICKNESS 
                            : x + w + LETTER_BORDER_THICKNESS]
                letter = self.smoothLetter(letter)

                if (y > img.shape[0] / 2):
                    bottom_letters.append(letter)
                    # cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
                else:
                    top_letters.append(letter)
                    # cv2.rectangle(img, (x, y), (x + w, y + h), 175, 2)

        return top_letters, bottom_letters

    def smoothLetter(self, letter):
        '''
        smooth the edges on the given letter image
        @param letter a binarized image of a letter'''

        letter = cv2.GaussianBlur(letter, (3, 3), 0)
        _, letter = cv2.threshold(letter, LETTER_COLOR_THRESHOLD, 255, cv2.THRESH_BINARY)

        kernel = np.ones((2,2))
        letter = cv2.morphologyEx(letter, cv2.MORPH_CLOSE, kernel)
        return letter

    def detectBoard(self, img):
        '''
        given a raw image, determine if a clue board can be found. If found, 
            perform a perspective transform to center the image on the clueboard
        @return True if board is found
        @return the transformed image
        '''
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, binarized = cv2.threshold(gray, BORDER_COLOR_THRESHOLD,255,cv2.THRESH_BINARY)

        erosionKernel = np.ones((3,3))
        eroded = cv2.erode(binarized, erosionKernel, iterations = 1)

        dilationKernel = np.ones((4,4))
        dilated = cv2.dilate(eroded, dilationKernel, iterations = 3)

        inverted = cv2.bitwise_not(dilated)
        contours, _ = cv2.findContours(inverted, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = contours[1:]
        # cv2.drawContours(gray, filtered_contours, -1, 255, 3)

        ret_img = img
        for cnt in filtered_contours:
            epsilon = 0.1 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) == 4:
                # cv2.polylines(gray, [approx], isClosed = True, color=255, thickness = 3)
                # contour_corners = np.float32(approx)
                contour_corners = self.sort_corners(approx)

                image_corners = np.float32([[0, 0], 
                                            [TRANSFORMED_CB_WIDTH - 1, 0], 
                                            [TRANSFORMED_CB_WIDTH - 1, TRANSFORMED_CB_HEIGHT - 1],
                                            [0, TRANSFORMED_CB_HEIGHT - 1]])
                
                matrix = cv2.getPerspectiveTransform(contour_corners, image_corners)
                ret_img = cv2.warpPerspective(gray, matrix, (TRANSFORMED_CB_WIDTH, TRANSFORMED_CB_HEIGHT))

        if (ret_img.all() == img.all()): return False, img
        return self.containsIcon(ret_img), ret_img

    def containsIcon(self, img):
        '''
        @return True if the Fizz Detective Icon can be found in img
        '''
        reference = cv2.imread(FIZZ_ICON_PATH, cv2.IMREAD_GRAYSCALE)
        sift = cv2.SIFT_create()
        kp_image, desc_image = sift.detectAndCompute(reference, None)

        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        kp_grayframe, desc_grayframe = sift.detectAndCompute(img, None)

        matches = flann.knnMatch(desc_image, desc_grayframe, k=2)
        good_points = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_points.append(m)

        if (len(good_points) < NUM_MATCHES_FOR_HOMOGRAPHY): return False

        query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        # Perspective transform
        h, w = reference.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        self.icon_corners = cv2.perspectiveTransform(pts, matrix)        
        # img = cv2.polylines(img, [np.int32(dst)], True, (255, 0, 0), 3)

        return True

    def sort_corners(self, corners):
        corners = [point[0] for point in corners]
        sorted_corners = sorted(corners, key=lambda x: x[0])
        
        left = sorted_corners[:2]
        right = sorted_corners[2:]

        left = sorted(left, key=lambda x: x[1])
        right = sorted(right, key=lambda x: x[1])
        
        return np.array([left[0], right[0], right[1], left[1]], dtype=np.float32)

