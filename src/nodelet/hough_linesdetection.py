#!/usr/bin/env python

# This code applies canny on the image stream and then detects for line on the image

import rospy, cv2, cv_bridge, numpy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point as line

class Follower:
  def __init__(self):
  	self.bridge = cv_bridge.CvBridge()
  	self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_cb)
  	self.line_pub = rospy.Publisher('/line_pub', line, queue_size=1)
  	self.line  = line()

  def image_cb(self, msg):
  	image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
  	frame = cv2.GaussianBlur(image, (5,5), 0)
  	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  	#list of HSV to detect: low_purple: 102,0,204 high_purple:255,255,255
  	low_black = numpy.array([102,0,204])
  	high_black = numpy.array([224,224,224])
  	mask = cv2.inRange(hsv,low_black,high_black)
  	edges = cv2.Canny(gray,100,150)
  	res = cv2.bitwise_and(image,image, mask= mask)

  	lines = cv2.HoughLinesP(edges, 1, numpy.pi/180, 175, maxLineGap = 175)
  	if lines is not None:
  		for line in lines:
  			# print line
  			x1, y1, x2, y2 = line[0]
  			# print line[0]
  			print 'x1, y1: ' + str(x1) + ', ' + str(y1)
  			print 'x2, y2: ' + str(x2) + ', ' + str(y2)
  			cv2.line(image,(x1,y1),(x2,y2),(0,255,0),3)
  			cv2.line(edges,(x1,y1),(x2,y2),(255,0,0),3)
  			self.line.x = line[0][0]
  			self.line.y = line[0][1]
  			self.line_pub.publish(self.line)
  			rospy.sleep(0.001)
  			self.line.x = line[0][2]
  			self.line.y = line[0][3]
  			self.line_pub.publish(self.line)
	cv2.imshow("Img Frame", image)
	cv2.imshow("Edges", edges)
	# cv2.imshow("res", res)
	cv2.waitKey(3)

rospy.init_node('Follower')
follower = Follower()
rospy.spin()

#END
