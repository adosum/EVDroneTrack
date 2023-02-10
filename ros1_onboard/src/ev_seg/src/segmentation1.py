#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 16:05:37 2021

@author: haixin
"""

import os
import time

import cv2
from filterpy.kalman import KalmanFilter
from numpy import empty

import torch
import numpy as np
# from ev_segment.unet_torch import UNet
from Models import U_Net
#from yolox.data.data_augment import preproc
#from yolox.data.datasets import COCO_CLASSES
#from yolox.exp import get_exp


import rospy

# from std_msgs.msg import Header
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3

from skimage.measure import label, regionprops

bridge = CvBridge()
model = U_Net(3,2)
ckpt_file = '/home/haixin/ros1/src/ev_seg/src/checkpoint.pth.tar'
ckpt = torch.load(ckpt_file)
model.load_state_dict(ckpt["state_dict"])
model = model.cuda() 
model.eval()
kalman = KalmanFilter(6,3)
kalman.x = np.array([1,1,30,0,0,0])
kalman.F = np.array([[1,0,0,1,0,0], [0,1,0,0,1,0], [0,0,1,0,0,1], [0, 0,0, 1,0, 0], [0, 0, 0,0,1, 0], [0, 0,0 ,0,0, 1]])
kalman.Q = 0.1 * np.eye(6) #Q_discrete_white_noise(dim=6, dt=0.1, var=0.13)
kalman.Q[2,2] *= 5
kalman.Q[5,5] *= 5
kalman.H = np.array([[1.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.],[0.,0.,1.,0.,0.,0.]])
kalman.P *= 1.
kalman.R *= 1.
pub_image = rospy.Publisher("seg/masks",Image, queue_size = 10)
pub_state = rospy.Publisher("/VisualState",Vector3, queue_size = 10)


def callback(data):
    img_rgb = bridge.imgmsg_to_cv2(data,"bgr8")
    img_rgb = np.array(img_rgb) 
    img_rgb = img_rgb.transpose((2, 0, 1))
    img_rgb = torch.tensor(img_rgb).unsqueeze(0).float().cuda()
    img_rgb = img_rgb/255
    mask = model(img_rgb)
    mask = torch.argmax(mask,1)
    mask = mask.squeeze(0).detach().cpu().numpy().astype(np.uint8)
    mask = mask*255
    contours,hierarchy = cv2.findContours(mask, 1, 2)
    try:
        contour = contours[0]
        moment = cv2.moments(contour)
        # Calculate area
        area = moment['m00']
        # Calculate centroid
        cx = int(moment['m10']/moment['m00'])
        cy = int(moment['m01']/moment['m00'])
    except:
        return
    pub_image.publish(bridge.cv2_to_imgmsg(mask,"mono8"))

    kalman.predict()
    kalman.update(np.array([cx,cy,area]))

    msg = Vector3()
    msg.x = kalman.x[0]
    msg.y = kalman.x[1]
    msg.z = kalman.x[2]
    pub_state.publish(msg)


def ros_main():
    rospy.init_node('ev_seg', anonymous=True)
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    rospy.Subscriber('/event_frame', Image, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == "__main__":
    ros_main()