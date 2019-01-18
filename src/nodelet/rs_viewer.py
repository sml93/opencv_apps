#!/usr/bin/env python

# This code applies canny on the image stream and then detects for line on the image

import rospy, cv2, cv_bridge, numpy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point as line
from cv_bridge import CvBridge, CvBridgeError



class Follower:
  def __init__(self):
  	self.bridge = cv_bridge.CvBridge()
  	self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_cb)
  	# self.line_pub = rospy.Publisher('/line_pub', line, queue_size=1)
  	# self.line  = line()

  def image_cb(self, msg):
    print("Received an image!")
    try:
      image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    except CvBridgeError, e:
      print(e)
    else:
      time = msg.header.stamp
      cv2.imwrite(''+str(time)+'.jpeg', image)
      rospy.sleep(0.5)

    cv2.imshow("Img Frame", image)
    cv2.waitKey(3)

rospy.init_node('Follower')
follower = Follower()
rospy.spin()

#END
