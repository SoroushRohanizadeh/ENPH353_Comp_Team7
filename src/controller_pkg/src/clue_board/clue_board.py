import cv2
import numpy as np
from collections import Counter
from skimage.metrics import structural_similarity as ssim

BORDER_COLOR_THRESHOLD_DARK = 15

TRANSFORMED_CB_WIDTH = 800
TRANSFORMED_CB_HEIGHT = 800

LETTER_COLOR_THRESHOLD_DARK = 70
LETTER_COLOR_THRESHOLD_LIGHT = 100
LETTER_COLOR_THRESHOLD_MED = 100
LETTER_HEIGHT_TOLERANCE = 5
MIN_LETTER_HEIGHT = 50
MIN_LETTER_WIDTH = 35
LETTER_BORDER_THICKNESS = 3

FIZZ_ICON_PATH = "clue_board/reference_images/fizz_detective.png"
NUM_MATCHES_FOR_HOMOGRAPHY = 8

LETTER_IMG_PATH = "clue_board/reference_images/"
LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']



class ClueBoard:

    def __init__(self):
        self.LIGHT_BOARD = False # 3, 6
        self.MEDIUM_BOARD = False # 4
        self.DARK_BOARD = True # 1, 2, 5, 7, 8

    def detectClueBoard(self, img):
        detected, transformed_img = self.detectBoard(img)
        if not detected: return False, 0, str()
        return True, self.parseBoard(transformed_img)
    
    def setFlags(self, num):
        if (num == 3 or 6):
            self.LIGHT_BOARD = True
            self.MEDIUM_BOARD = False
            self.DARK_BOARD = False
        elif num == 4:
            self.LIGHT_BOARD = False
            self.MEDIUM_BOARD = True
            self.DARK_BOARD = False
        elif num == 1 or 2 or 5 or 7 or 8:
            self.LIGHT_BOARD = False
            self.MEDIUM_BOARD = False
            self.DARK_BOARD = True

    def detectClueBoard_Debug(self, img):
        '''
        used only during development
        '''
        detected, transformed_img = self.detectBoard(img)
        if not detected: return img

        top, bottom = self.highlightLetters(transformed_img)
        top_msg, bottom_msg = self.parseBoard(transformed_img)
        return transformed_img
        # return self.highlightLetters(transformed_img)
    
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

        top_msg = ''.join(top_chars)
        bottom_msg = ''.join(bottom_chars)
        print('Top: ' + ''.join(top_chars) + " Bottom: " + ''.join(bottom_chars))
        return top_msg, bottom_msg

    def imgDifference(self, img1, img2):
        height = max(img1.shape[0], img2.shape[0])
        width = max(img1.shape[1], img2.shape[1])

        img1 = self.padImg(img1, height, width)
        img2 = self.padImg(img2, height, width)

        return np.mean((img1 - img2) ** 2)
        # return ssim(img1, img2, full = True)[0]

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
        diffs = []
        for ref_letter in LETTERS:
            ref_img = cv2.imread(LETTER_IMG_PATH + ref_letter + ".png", cv2.IMREAD_GRAYSCALE)
            diffs.append(self.imgDifference(letter, ref_img))

        guess = LETTERS[diffs.index(min(diffs))]
        if guess == "G" or guess == "C":
            return self.cOrG(letter)
        if guess == 'E' or guess == 'F':
            return self.eOrF(letter)
        if guess == 'I' or guess == 'I':
            return self.iOrOne(letter)
        return guess

    def iOrOne(self, letter):
        
        cropped = self.topHalf(letter)

        i = cv2.imread(LETTER_IMG_PATH + "I.png", cv2.IMREAD_GRAYSCALE)
        i = self.topHalf(i)

        one = cv2.imread(LETTER_IMG_PATH + "1.png", cv2.IMREAD_GRAYSCALE)
        one = self.topHalf(one)

        i_diff = self.imgDifference(i, cropped)
        one_diff = self.imgDifference(one, cropped)

        if (i_diff > one_diff):
            return "1"
        else:
            return "I"

    def topHalf(self, letter):
        height, width = letter.shape[:2]

        crop_height = height // 2
        return letter[0:height - crop_height, :]

    def eOrF(self, letter):
        
        cropped = self.bottomHalf(letter)

        e = cv2.imread(LETTER_IMG_PATH + "E.png", cv2.IMREAD_GRAYSCALE)
        e = self.bottomHalf(e)

        f = cv2.imread(LETTER_IMG_PATH + "F.png", cv2.IMREAD_GRAYSCALE)
        f = self.bottomHalf(f)

        e_diff = self.imgDifference(e, cropped)
        f_diff = self.imgDifference(f, cropped)

        if (e_diff > f_diff):
            return "F"
        else:
            return "E"

    def bottomHalf(self, letter):
        height, width = letter.shape[:2]

        crop_height = height // 2
        return letter[height - crop_height:height, :]

    def cOrG(self, letter):
        cropped = self.bottomRight(letter)

        g = cv2.imread(LETTER_IMG_PATH + "G.png", cv2.IMREAD_GRAYSCALE)
        g = self.bottomRight(g)

        c = cv2.imread(LETTER_IMG_PATH + "C.png", cv2.IMREAD_GRAYSCALE)
        c = self.bottomRight(c)

        g_diff = self.imgDifference(g, cropped)
        c_diff = self.imgDifference(c, cropped)

        if (c_diff > g_diff):
            return "G"
        else:
            return "C"

    def bottomRight(self, letter):
        height, width = letter.shape[:2]

        crop_height = height // 2
        crop_width = width // 2
        return letter[height - crop_height:height, width - crop_width:width]

    def highlightLetters(self, img):
        '''
        Identify and highlight the letters in the image
        '''
        if (self.LIGHT_BOARD):
            ret, binarized = cv2.threshold(img, LETTER_COLOR_THRESHOLD_LIGHT,255,cv2.THRESH_BINARY)
        elif (self.DARK_BOARD):
            ret, binarized = cv2.threshold(img, LETTER_COLOR_THRESHOLD_DARK,255,cv2.THRESH_BINARY) 
        elif (self.MEDIUM_BOARD):
            ret, binarized = cv2.threshold(img, LETTER_COLOR_THRESHOLD_MED,255,cv2.THRESH_BINARY) 

        dilationKernel = np.ones((2,2))
        dilated = cv2.dilate(binarized, dilationKernel, iterations = 1)

        inverted = cv2.bitwise_not(dilated)

        cv2.fillPoly(inverted, [np.int32(self.icon_corners)], color = 0)

        contours, _ = cv2.findContours(inverted, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        contours = sorted(contours, key = cv2.contourArea, reverse = True)

        if (self.DARK_BOARD or self.MEDIUM_BOARD):
            contours = contours[2:] # remove the border contours
        # cv2.drawContours(img, contours, -1, 255, 3)
        contours = [cnt for cnt in contours if cv2.boundingRect(cnt)[3] > MIN_LETTER_HEIGHT]
        contours = self.filterInternalContours(contours)

        contours = sorted(contours, key = lambda contour: cv2.boundingRect(contour)[0]) #sort by x location
        
        # detect for cases where 2 nearby letters are mistaken as 1 contour
        widths = [cv2.boundingRect(contour)[2] for contour in contours]
        width_counter = Counter(widths)
        most_common_width, count = width_counter.most_common(1)[0]

        top_letters = []
        bottom_letters = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            letter = dilated[y - LETTER_BORDER_THICKNESS 
                        : y + h + LETTER_BORDER_THICKNESS, 
                        x - LETTER_BORDER_THICKNESS 
                        : x + w + LETTER_BORDER_THICKNESS]
            letter = self.smoothLetter(letter)

            if (y > img.shape[0] / 2):
                if (w > 1.5 * most_common_width):
                    bottom_letters.append(letter[:, :w // 2])
                    bottom_letters.append(letter[:, w // 2: ])
                    continue

                bottom_letters.append(letter)
                cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
            else:
                if (w > 1.5 * most_common_width):
                    top_letters.append(letter[:, :w // 2])
                    top_letters.append(letter[:, w // 2: ])
                    continue
                
                top_letters.append(letter)
                cv2.rectangle(img, (x, y), (x + w, y + h), 175, 2)

        return top_letters, bottom_letters

    def filterInternalContours(self, contours):
        bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

        outer_contours = []
        for i, box1 in enumerate(bounding_boxes):
            x1, y1, w1, h1 = box1
            inside = False
            
            for j, box2 in enumerate(bounding_boxes):
                if i != j:
                    x2, y2, w2, h2 = box2
                    if x1 > x2 and y1 > y2 and (x1 + w1) < (x2 + w2) and (y1 + h1) < (y2 + h2):
                        inside = True
                        break
            
            if not inside:
                outer_contours.append(contours[i])
        return outer_contours

    def smoothLetter(self, letter):
        '''
        smooth the edges on the given letter image
        @param letter a binarized image of a letter'''

        letter = cv2.GaussianBlur(letter, (3, 3), 0)
        if (self.LIGHT_BOARD):
            _, letter = cv2.threshold(letter, LETTER_COLOR_THRESHOLD_LIGHT, 255, cv2.THRESH_BINARY)
        elif (self.DARK_BOARD):
            _, letter = cv2.threshold(letter, LETTER_COLOR_THRESHOLD_DARK, 255, cv2.THRESH_BINARY)
        elif (self.MEDIUM_BOARD):
            _, letter = cv2.threshold(letter, LETTER_COLOR_THRESHOLD_DARK, 255, cv2.THRESH_BINARY)

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

        if (self.LIGHT_BOARD):
            lower_blue = (190, 90, 90)
            upper_blue = (210, 110, 110)
            binarized = cv2.bitwise_not(cv2.inRange(img, lower_blue, upper_blue))
        elif (self.DARK_BOARD):
            ret, binarized = cv2.threshold(gray, BORDER_COLOR_THRESHOLD_DARK,255,cv2.THRESH_BINARY)
        elif (self.MEDIUM_BOARD):
            lower_blue = (110, 10, 10)
            upper_blue = (130, 30, 30)
            binarized = cv2.bitwise_not(cv2.inRange(img, lower_blue, upper_blue))

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

        try:
            matches = flann.knnMatch(desc_image, desc_grayframe, k=2)
        except Exception as e:
            return False
        
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

