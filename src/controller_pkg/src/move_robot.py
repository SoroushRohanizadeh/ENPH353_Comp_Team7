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

        self.lower_blue = lower_blue2
        self.upper_blue = upper_blue2

        if DEBUG:
            self.debug = rospy.Publisher('/image_debug', Image, queue_size = 1)

        self.curr_state = "TP_1"

        self.states = {

            "LF": self.line_follow_state,
            "LF_DIRT": self.line_follow_dirt_state,
            "ANDY_WAIT": self.save_andy_state,
            "ANDY_GO": self.go_andy_state,
            "TP_1": self.tp_1_state,
            # "TP_2": self.tp_2_state,
            "CB_1": self.cb_1_state,
            "CB_2": self.cb_2_state,
            "CB_3": self.cb_3_state,
            "CB_4": self.cb_4_state,
            "CB_5": self.cb_5_state,
        }

    def run(self):

        while not rospy.Time.now().to_sec() > 0:
            rospy.sleep(0.1)

        while not rospy.is_shutdown():
            self.cmd_vel.publish(self.moveCommand) # should only be called here
            self.rate.sleep()

    def cameraCallback(self, img):
        
        self.states[self.curr_state](img)

        if DEBUG:
            img = self.cb.detectClueBoard_Debug(self.bridge.imgmsg_to_cv2(img, desired_encoding='bgr8'))
            self.debug.publish(self.bridge.cv2_to_imgmsg(img))
        else:
            ret, num, msg = self.cb.detectClueBoard(self.bridge.imgmsg_to_cv2(img, desired_encoding='bgr8'))
            if (ret):
                self.publishClue(num, msg)

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

    def cb_2_state(self,img):
        x,yaw = lf.scan_cb(self,img, self.lower_blue, self.upper_blue)
        self.setMotion(x,yaw)

    def cb_3_state(self,img):
        x,yaw = lf.scan_cb(self,img, self.lower_blue, self.upper_blue)
        self.setMotion(x,yaw)

    def cb_4_state(self,img):
        x,yaw = lf.scan_cb(self,img, self.lower_blue, self.upper_blue)
        self.setMotion(x,yaw)

    def cb_5_state(self,img):
        self.setMotion(0,0)
        rospy.sleep(0.5)
        self.setMotion(0,2.0)
        rospy.sleep(1.0)
        self.setMotion(0,0)
        rospy.sleep(1.0)
        self.setMotion(1.0,-1.0)
        rospy.sleep(2.3)
        self.setMotion(0,0)
        rospy.sleep(5.0)
        self.curr_state = "LF_DIRT"
    def tp_1_state(self,img):

        self.setMotion(0,0)

        position_1 = [0.56, 0, 0.1, np.pi/2, 0, np.pi, np.pi]
        self.respawn(position_1)
        rospy.sleep(1)
        self.setMotion(-2,0)
        rospy.sleep(1.6)
        self.setMotion(0,0)
        rospy.sleep(2.0)
        self.curr_state = "LF_DIRT"
if __name__ == '__main__':
    try:
        node = ControlNode()
        node.run()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
