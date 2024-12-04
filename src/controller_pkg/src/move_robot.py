#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState

from clue_board.clue_board import ClueBoard
import line_follow.line_follow as lf
import cv2
import numpy as np

DEBUG = True

lower_blue1 = np.array([80,0,0])
upper_blue1 = np.array([120,20,20])

lower_blue2 = np.array([190,90,90])
upper_blue2 = np.array([210,110,110])

lower_blue3 = np.array([110,10,10])
upper_blue3 = np.array([130,30,30])

class ControlNode:

    def __init__(self):
        rospy.init_node('move_robot', anonymous=True)
        self.bridge = CvBridge()
        self.moveCommand = Twist()
        self.string = String()
        self.cb = ClueBoard()

        self.state_pub = rospy.Publisher('/line_follower/state', String, queue_size=1)
        self.cmd_vel = rospy.Publisher('B1/cmd_vel', Twist, queue_size = 1)
        self.score_tracker = rospy.Publisher('/score_tracker', String, queue_size = 1)
        self.cam = rospy.Subscriber('/B1/rrbot/camera1/image_raw', Image, self.cameraCallback)
        self.rate = rospy.Rate(1000)

        self.lower_blue = lower_blue1
        self.upper_blue = upper_blue1

        if DEBUG:
            self.debug = rospy.Publisher('/image_debug', Image, queue_size = 1)

        self.curr_state = "LF"

        self.states = {

            "LF": self.line_follow_state,
            "LF_DIRT": self.line_follow_dirt_state,
            "ANDY_WAIT": self.save_andy_state,
            "ANDY_GO": self.go_andy_state,
            "TP_1": self.tp_1_state,
            "TP_2": self.tp_2_state,
            "CB_1": self.cb_1_state,
            "CB_2": self.cb_2_state,
            "CB_3": self.cb_3_state,
            "CB_4": self.cb_4_state,
            "CB_5": self.cb_5_state,
            "CB_6": self.cb_6_state,
            "CB_7": self.cb_7_state,
        }
        self.current_board_num = 1

    def run(self):

        self.startTimer()
        # while not rospy.Time.now().to_sec() > 0:
        #     rospy.sleep(0.1)

        while not rospy.is_shutdown():
            self.cmd_vel.publish(self.moveCommand) # should only be called here
            self.rate.sleep()

    def cameraCallback(self, img):
        
        self.states[self.curr_state](img)
        self.updateBoardNumber()
        print("Looking for: ", self.current_board_num)

        # if DEBUG:
        #     img = self.cb.detectClueBoard_Debug(self.current_board_num, self.bridge.imgmsg_to_cv2(img, desired_encoding='bgr8'))
        #     self.debug.publish(self.bridge.cv2_to_imgmsg(img))
        # else:
        #     ret, _, msg = self.cb.detectClueBoard(self.current_board_num, self.bridge.imgmsg_to_cv2(img, desired_encoding='bgr8'))
        #     if (ret):
        #         self.publishClue(self.current_board_num, msg)

    def updateBoardNumber(self):
        if self.curr_state == "CB_1":
            self.current_board_num = 1
        elif self.curr_state == "CB_2":
            self.current_board_num = 2
        elif self.curr_state == "CB_3":
            self.current_board_num = 3
        elif self.curr_state == "TP_1" or self.curr_state == "CB_4":
            self.current_board_num = 4
        elif self.curr_state == "CB_5":
            self.current_board_num = 5
        elif self.curr_state == "CB_6":
            self.current_board_num = 6
        elif self.curr_state == "CB_7" or self.curr_state == "TP_2":
            self.current_board_num = 7

    def publishClue(self, num, msg):
        self.score_tracker.publish("Team7,password," + num + "," + msg)

    def startTimer(self):
        self.score_tracker.publish("Team7,password,0,NA")

    def stopTimer(self):
        self.score_tracker.publish("Team7,password,-1,NA")

    def setMotion(self, x, yaw = 0):
        self.moveCommand.linear.x = float(x)
        self.moveCommand.angular.z = float(yaw)

    def stop(self):
        self.setMotion(0, 0)

    def respawn(self, position):

        msg = ModelState()
        msg.model_name = 'B1'

        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]
        msg.pose.orientation.x = position[3]
        msg.pose.orientation.y = position[4]
        msg.pose.orientation.z = position[5]
        msg.pose.orientation.w = position[6]

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(msg)

        except rospy.ServiceException:
            print ("Service call failed")

    def line_follow_state(self,img):

        x,yaw = lf.line_follow(self, img, self.lower_blue, self.upper_blue)
        # x,yaw = lf.line_follow_leaves(self,img)
        x,yaw = lf.acc_comms(x,yaw)
        # x,yaw = (0,0)
        self.setMotion(x, yaw)

    def line_follow_dirt_state(self,img):

        x,yaw = lf.line_follow_leaves(self,img, self.lower_blue, self.upper_blue)
        x,yaw = lf.acc_comms(x,yaw)
        self.setMotion(x, yaw)

    def save_andy_state(self,img):

        x,yaw = lf.save_andy(self,img)
        self.setMotion(x, yaw)

    def go_andy_state(self,img):
        x,yaw = lf.go_andy(self,img)
        self.setMotion(x, yaw)
        rospy.sleep(2)

    def cb_1_state(self,img):
        x,yaw = lf.scan_cb(self,img, self.lower_blue, self.upper_blue)
        self.setMotion(x,yaw)
        yay, extra, result = self.cb.detectClueBoard(1,self.bridge.imgmsg_to_cv2(img, desired_encoding='bgr8'))
        if yay:
            self.setMotion(0,-1)
            rospy.sleep(1)
            self.curr_state = "LF"
            self.publishClue(1, result)
        # else: #only because cb broken
        #     self.curr_state="LF"

    def cb_2_state(self,img):
        x,yaw = lf.scan_cb(self,img, self.lower_blue, self.upper_blue)
        self.setMotion(x,yaw)
        yay, extra, result = self.cb.detectClueBoard(2,self.bridge.imgmsg_to_cv2(img, desired_encoding='bgr8'))
        if yay:
            self.setMotion(0,1)
            rospy.sleep(1)
            self.curr_state = "LF"
            self.lower_blue = lower_blue2
            self.upper_blue = upper_blue2
            self.publishClue(2, result)
        # else: #only because cb broken
        #     self.setMotion(0,1.7)
        #     rospy.sleep(1)
        #     self.curr_state="LF"
        #     self.lower_blue = lower_blue2
        #     self.upper_blue = upper_blue2

    def cb_3_state(self,img):
        x,yaw = lf.scan_cb(self,img, self.lower_blue, self.upper_blue)
        self.setMotion(x,yaw)
        yay, extra, result = self.cb.detectClueBoard(3,self.bridge.imgmsg_to_cv2(img, desired_encoding='bgr8'))
        if yay:
            self.setMotion(0,0)
            rospy.sleep(0.5)
            print("TP_1")
            self.curr_state = "TP_1"
            self.lower_blue = lower_blue3
            self.upper_blue = upper_blue3
            self.publishClue(3, result)
        # else: #only because cb broken
        #     self.setMotion(0,0)
        #     # rospy.sleep(0.5)
        #     print("TP_1")
        #     self.curr_state="TP_1"
        #     self.lower_blue = lower_blue3
        #     self.upper_blue = upper_blue3

    def cb_4_state(self,img):
        x,yaw = lf.scan_cb(self,img, self.lower_blue, self.upper_blue)
        self.setMotion(x,yaw)
        yay, extra, result = self.cb.detectClueBoard(4,self.bridge.imgmsg_to_cv2(img, desired_encoding='bgr8'))
        if yay:
            self.setMotion(0,1)
            rospy.sleep(1)
            self.curr_state = "LF_DIRT"
            self.lower_blue = lower_blue2
            self.upper_blue = upper_blue2
            self.publishClue(4, result)
        # else: #only because cb broken
        #     self.setMotion(2,0)
        #     rospy.sleep(1.0)
        #     self.curr_state="LF_DIRT"
        #     self.lower_blue = lower_blue2
        #     self.upper_blue = upper_blue2

    def cb_5_state(self,img):
        self.setMotion(0,0)
        rospy.sleep(0.5)
        self.setMotion(0,2.0)
        rospy.sleep(1.5)
        self.setMotion(0,0)
        rospy.sleep(1.0)
        self.setMotion(1.0,-1.73)
        rospy.sleep(2.8)
        self.setMotion(2.0,0)
        rospy.sleep(3.0)
        self.curr_state = "LF_DIRT"
    
    def cb_6_state(self,img):
        x,yaw = lf.scan_cb(self,img, self.lower_blue, self.upper_blue)
        self.setMotion(x,yaw)
        yay, extra, result = self.cb.detectClueBoard(6,self.bridge.imgmsg_to_cv2(img, desired_encoding='bgr8'))
        if yay:
            self.setMotion(0,0)
            rospy.sleep(0.5)
            print("TP_2")
            self.curr_state = "TP_2"
            self.lower_blue = lower_blue1
            self.upper_blue = upper_blue1
            self.publishClue(6, result)
        # else: #only because cb broken
        #     self.setMotion(0,0)
        #     rospy.sleep(0.5)
        #     print("TP_2")
        #     self.curr_state="TP_2"
        #     self.lower_blue = lower_blue1
        #     self.upper_blue = upper_blue1

    def cb_7_state(self,img):
        x,yaw = lf.scan_cb(self,img, self.lower_blue, self.upper_blue)
        self.setMotion(x,yaw)
        yay, extra, result = self.cb.detectClueBoard(6,self.bridge.imgmsg_to_cv2(img, desired_encoding='bgr8'))
        if yay:
            self.publishClue(7, result)
            self.stopTimer()

    def tp_1_state(self,img):

        self.setMotion(0,0)

        position_1 = [0.56, 0, 0.1, np.pi/2, 0, np.pi, np.pi]
        self.respawn(position_1)
        rospy.sleep(1)
        self.setMotion(-2,0)
        rospy.sleep(1.6)
        self.setMotion(0,0)
        rospy.sleep(2.0)
        self.lower_blue = lower_blue3
        self.upper_blue = upper_blue3
        print("CB_4")
        self.curr_state = "CB_4"

    def tp_2_state(self,img):

        self.setMotion(0,0)
        position_2 = [-4, -2.3, 0.1, np.pi/2, 0, np.pi, np.pi*20]
        self.respawn(position_2)
        rospy.sleep(1)
        self.setMotion(-0.3,2.0)
        rospy.sleep(1.0)
        self.setMotion(0,0)
        rospy.sleep(2.0)
        print("CB_7")
        self.curr_state = "CB_7"

if __name__ == '__main__':
    try:
        node = ControlNode()
        node.run()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
