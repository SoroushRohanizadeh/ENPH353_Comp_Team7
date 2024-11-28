#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

import clue_board.clue_board as cb
import line_follow.line_follow as lf

class ControlNode:

    def __init__(self):
        rospy.init_node('move_robot', anonymous=True)

        self.bridge = CvBridge()
        self.moveCommand = Twist()
        self.string = String()

        self.cmd_vel = rospy.Publisher('B1/cmd_vel', Twist, queue_size = 1)
        self.score_tracker = rospy.Publisher('/score_tracker', String, queue_size = 1)
        self.cam = rospy.Subscriber('/B1/pi_camera/image_raw', Image, self.cameraCallback)
        self.rate = rospy.Rate(1)

    def run(self):

        while not rospy.Time.now().to_sec() > 0:
            rospy.sleep(0.1)

        rospy.sleep(1.0)
        self.startTimer()
        rospy.sleep(1.0)
        self.setMotion(1)
        self.cmd_vel.publish(self.moveCommand)

        rospy.sleep(5.0)
        self.stop()
        self.cmd_vel.publish(self.moveCommand)
        self.stopTimer()

        while not rospy.is_shutdown():
            self.cmd_vel.publish(self.moveCommand) # should only be called here
            self.rate.sleep()

    def cameraCallback(self, img):
        self.setMotion(lf.line_follow(img))
        msg = cb.detectClueBoard(self.bridge.imgmsg_to_cv2(img, desired_encoding='bgr8'))

    def startTimer(self):
        self.score_tracker.publish("Team7,password,0,NA")

    def stopTimer(self):
        self.score_tracker.publish("Team7,password,-1,NA")

    def setMotion(self, x, yaw = 0):
        self.moveCommand.linear.x = x
        self.moveCommand.angular.z = yaw

    def stop(self):
        self.setMotion(0, 0)

if __name__ == '__main__':
    try:
        node = ControlNode()
        print("node init")
        node.run()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
