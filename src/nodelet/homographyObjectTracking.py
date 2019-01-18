#!/usr/bin/env python

import cv2, rospy, cv_bridge
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point as pixels
 

class OT:
    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_cb)
        self.pixel_pub = rospy.Publisher('obj_points', pixels, queue_size=1)
        self.pixels = pixels()

    def image_cb(self, msg):
        img = cv2.imread("linkway.jpg", cv2.IMREAD_GRAYSCALE) # queryimage
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')


        ###Features###
        sift = cv2.xfeatures2d.SIFT_create() #scale-invariant feature transform
        surf = cv2.xfeatures2d.SURF_create() #speeded up robust features
        orb = cv2.ORB_create(nfeatures=1500)

        ###Feature Matching###
        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)


        ###Applying Features on Still Image###
        kp_image, desc_image = sift.detectAndCompute(img, None)
        # img = cv2.drawKeypoints(img, kp_image, img)
        

        ###Converting image stream to GRAY###
        grayframe = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #trainimage

        ###Applying Features on Image Stream###
        kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)
        # grayframe = cv2.drawKeypoints(grayframe, kp_grayframe, grayframe)
        matches = flann.knnMatch(desc_image, desc_grayframe, k=2)


        good_points = []
        for m, n in matches: #m = query image, n is object from grayframe
            if m.distance < 0.5 * n.distance:
                good_points.append(m)


        # opimg = cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_points, grayframe)

        ###Homography###
        if len(good_points) > 10:
            query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)

            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()

            #Perspective transform
            h, w = img.shape
            pts = np.float32([[0, 0], [0, h], [w,h], [w,0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)

            ###Drawing an encapsulation on top of the image stream###
            homography = cv2.polylines(image, [np.int32(dst)], True, (0,255,0), 3)

            ###Showing the output of the image stream with encapsulation###
            cv2.imshow("Homography", homography)

            # print dst
            # print pixels
            for pixels in dst:
                x1, y1 = pixels[[0,0]]
                self.pixels.x = x1[[0]]
                self.pixels.y = y1[[1]]
                self.pixel_pub.publish(self.pixels)
                # rospy.sleep(0.001)
        else:
            ###if the image is not in the FOV, show only image stream with no encapsulation###
            cv2.imshow("Homography", image)



        ###Showing output of images###
        # cv2.imshow("Img Frame", img)
        # cv2.imshow("grayFrame", grayframe)
        # cv2.imshow("Output Image", opimg)
        cv2.waitKey(1)

        

rospy.init_node('OT')
ot = OT()
rospy.spin()

#END