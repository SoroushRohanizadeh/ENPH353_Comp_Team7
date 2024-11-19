#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np

class ControlNode:

    def __init__(self):
        rospy.init_node('move_robot', anonymous=True)

        self.bridge = CvBridge()
        self.move = Twist()

        self.pub = rospy.Publisher('B1/cmd_vel', Twist, queue_size=1)
        self.rate = rospy.Rate(1)

        self.move.linear.x = 0.5
        self.pub.publish(self.move)

    

    def run(self):
        while not rospy.is_shutdown():
            self.pub.publish(self.move)
            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = ControlNode()
        print("node init")
        node.run()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
