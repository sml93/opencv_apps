#! usr/bin/env python

import cv2, rospy, cv_bridge
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point as pixels
 

class BG:
    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_cb)
        self.pixel_pub = rospy.Publisher('obj_points', pixels, queue_size=1)
        self.pixels = pixels()

    def image_cb(self, msg):
        img = cv2.imread("linkway.jpg", cv2.IMREAD_GRAYSCALE) # queryimage
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        cv2.imshow("Img Frame", img)

rospy.init_node('BG')
bg = BG()
rospy.spin()

#END