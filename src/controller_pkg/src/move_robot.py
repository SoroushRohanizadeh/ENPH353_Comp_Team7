#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np


class ControlNode:

    def __init__(self):
        rospy.init_node('move_robot', anonymous=True)

        self.bridge = CvBridge()
        self.move = Twist()
        self.string = String()

        self.cmd_vel = rospy.Publisher('B1/cmd_vel', Twist, queue_size = 1)
        self.score_tracker = rospy.Publisher('/score_tracker', String, queue_size = 1)
        self.rate = rospy.Rate(1)

        self.move.linear.x = 0.5

    

    def run(self):

        self.string = "7, 7, 0, hello"
        self.score_tracker.publish(self.string)
        print("pushed" + self.string)
        while not rospy.is_shutdown():
            self.cmd_vel.publish(self.move)
            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = ControlNode()
        print("node init")
        node.run()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
