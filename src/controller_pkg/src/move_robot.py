#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from clue_board.clue_board import ClueBoard
import line_follow.line_follow as lf
import cv2

DEBUG = True

class ControlNode:

    def __init__(self):
        rospy.init_node('move_robot', anonymous=True)

        self.bridge = CvBridge()
        self.moveCommand = Twist()
        self.string = String()
        self.cb = ClueBoard()

        self.cmd_vel = rospy.Publisher('B1/cmd_vel', Twist, queue_size = 1)
        self.score_tracker = rospy.Publisher('/score_tracker', String, queue_size = 1)
        self.cam = rospy.Subscriber('/B1/rrbot/camera1/image_raw', Image, self.cameraCallback)
        self.rate = rospy.Rate(100)

        if DEBUG:
            self.debug = rospy.Publisher('/image_debug', Image, queue_size = 1)

    def run(self):

        # while not rospy.Time.now().to_sec() > 0:
        #     rospy.sleep(0.1)

        # rospy.sleep(1.0)
        # self.startTimer()
        # rospy.sleep(1.0)
        # self.setMotion(1)
        # self.cmd_vel.publish(self.moveCommand)

        # rospy.sleep(5.0)
        # self.stop()
        # self.cmd_vel.publish(self.moveCommand)
        # self.stopTimer()

        while not rospy.is_shutdown():
            self.cmd_vel.publish(self.moveCommand) # should only be called here
            self.rate.sleep()

    def cameraCallback(self, img):
        x, yaw = lf.line_follow(img)
        self.setMotion(x, yaw)

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


if __name__ == '__main__':
    try:
        node = ControlNode()
        node.run()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
