#!/usr/bin/env python

#This code applies SIFT/SURF/ORB on a image or image stream and then detects for features

import rospy, cv2, cv_bridge
import numpy as numpy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

class FD:
	def __init__(self):
		self.bridge = cv_bridge.CvBridge()
		self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_cb)

	def image_cb(self, msg):
		print("Received an image!")
		try:
			image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
		except CvBridgeError, e:
			print(e)
		else:
			# img = cv2.imread("the_book_thief.jpg", cv2.IMREAD_GRAYSCALE)
			
			sift = cv2.xfeatures2d.SIFT_create()
			
			surf = cv2.xfeatures2d.SURF_create()

			orb = cv2.ORB_create(nfeatures=1500)

			keypoints, descriptors = surf.detectAndCompute(image, None)
			feat =  cv2.drawKeypoints(image, keypoints, None)

		cv2.imshow("Img Frame", image)
		cv2.imshow("Features", feat)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

rospy.init_node('FD')
featdect = FD()
rospy.spin()

#END