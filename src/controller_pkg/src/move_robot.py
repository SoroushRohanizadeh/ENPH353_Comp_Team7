#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

import clue_board as cb

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
        self.move(1)
        rospy.sleep(5.0)
        self.stop()
        self.stopTimer()

        while not rospy.is_shutdown():
            self.cmd_vel.publish(self.moveCommand)
            self.rate.sleep()

    def cameraCallback(self, img):
        msg = cb.detectClueBoard(self.bridge.imgmsg_to_cv2(img, desired_encoding='bgr8'))

    def startTimer(self):
        self.score_tracker.publish("Team7,password,0,NA")

    def stopTimer(self):
        self.score_tracker.publish("Team7,password,-1,NA")

    def move(self, x, yaw = 0):
        self.moveCommand.linear.x = x
        self.moveCommand.angular.z = yaw
        self.cmd_vel.publish(self.moveCommand)

    def stop(self):
        self.move(0, 0)

if __name__ == '__main__':
    try:
        node = ControlNode()
        print("node init")
        node.run()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
