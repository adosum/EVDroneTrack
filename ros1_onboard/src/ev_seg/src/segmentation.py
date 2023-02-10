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
from Models import U_Net,U_Net_hybrid  
#from yolox.data.data_augment import preproc
#from yolox.data.datasets import COCO_CLASSES
#from yolox.exp import get_exp


import rospy

# from std_msgs.msg import Header
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3

from skimage.measure import label, regionprops
# from rclpy.qos import qos_profile_sensor_data
# qos_profile_sensor_data.depth = 1



class seg_ros():
    def __init__(self) -> None:

        # ROS2 init
        # super().__init__('seg_ros')

        self.setting_seg_config()
        
        self.bridge = CvBridge()
        
#        self.pub = self.create_publisher(BoundingBoxes,"seg/bounding_boxes", 10)
        self.pub_image = rospy.Publisher("seg/masks",Image, queue_size = 1)
        self.pub_state = rospy.Publisher("/VisualState",Vector3, queue_size = 1)
        self.sub = rospy.Subscriber("/event_frame",Image,self.imageflow_callback, queue_size = 1)
    def setting_seg_config(self) -> None:
        # set environment variables for distributed training
        
        # ==============================================================

        # WEIGHTS_PATH = ''

        # ckpt_file_description = ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description='CKPT File Path')

        # self.declare_parameter('ckpt_file', WEIGHTS_PATH, ckpt_file_description)
#        self.declare_parameter('img_size', 640)
#        self.declare_param1eter('image_size/width', 640)
#        self.declare_parameter('image_size/height', 480)

        # self.ckpt_file = self.get_parameter('ckpt_file').value

#        ckpt_file = self.get_parameter('ckpt_file').value
        #ckpt = torch.load(self.ckpt_file)
        self.model = U_Net(3,2)
        ckpt_file = '/home/haixin/ros1/src/ev_seg/src/model_best.pth.tar'
        ckpt = torch.load(ckpt_file)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model = self.model.cuda() 
        self.model.eval()
        self.kalman = KalmanFilter(6,3)
        self.kalman.x = np.array([1,1,30,0,0,0])
        self.kalman.F = np.array([[1,0,0,1,0,0], [0,1,0,0,1,0], [0,0,1,0,0,1], [0, 0,0, 1,0, 0], [0, 0, 0,0,1, 0], [0, 0,0 ,0,0, 1]])
        self.kalman.Q = 0.1 * np.eye(6) #Q_discrete_white_noise(dim=6, dt=0.1, var=0.13)
        self.kalman.Q[2,2] *= 5
        self.kalman.Q[5,5] *= 5
        self.kalman.H = np.array([[1.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.],[0.,0.,1.,0.,0.,0.]])
        self.kalman.P *= 1.
        self.kalman.R *= 1.


#        self.predictor = Predictor(model, exp, COCO_CLASSES, decoder)

#    def yolox2bboxes_msgs(self, bboxes, scores, cls, cls_names, img_header:Header):
#        bboxes_msg = BoundingBoxes()
#        bboxes_msg.header = img_header
#        i = 0
#        for bbox in bboxes:
#            one_box = BoundingBox()
#            one_box.xmin = int(bbox[0])
#            one_box.ymin = int(bbox[1])
#            one_box.xmax = int(bbox[2])
#            one_box.ymax = int(bbox[3])
#            one_box.probability = float(scores[i])
#            one_box.class_id = str(cls_names[int(cls[i])])
#            bboxes_msg.bounding_boxes.append(one_box)
#            i = i+1
#        
#        return bboxes_msg
    def mask_to_bbox(self,mask):
        lbl_0 = label(mask) 
        props = regionprops(lbl_0)
        bbox = []
        for prop in props:
            boxA = prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2]
            bbox.append(boxA)
        return bbox



    def imageflow_callback(self,msg) -> None:
        # self.timestamp = rospy.get_time()
        img_rgb = self.bridge.imgmsg_to_cv2(msg,"bgr8")
        img_rgb = cv2.resize(img_rgb, (256, 256))
        img_rgb = np.array(img_rgb)
        img_rgb = img_rgb.transpose((2, 0, 1))
        img_rgb = torch.tensor(img_rgb).unsqueeze(0).float().cuda()
        img_rgb = img_rgb/255

        # img_rgb = self.bridge.imgmsg_to_cv2(msg,"bgr8")
        # img_rgb = cv2.resize(img_rgb, (256, 256))
        # img_rgb = np.array(img_rgb) 
        # img_rgb = img_rgb.transpose((2, 0, 1))
        # img_rgb = torch.tensor(img_rgb).unsqueeze(0).float().cuda()
        # img_rgb = img_rgb/255


        mask = self.model(img_rgb)
        mask = torch.argmax(mask,1)
        mask = mask.squeeze(0).detach().cpu().numpy().astype(np.uint8)
        mask = mask*255
        contours,hierarchy = cv2.findContours(mask, 1, 2)
        # try:
        #     contour = contours[0]
        #     moment = cv2.moments(contour)
        #     # Calculate area
        #     area = moment['m00']
        #     # Calculate centroid
        #     cx = int(moment['m10']/moment['m00'])
        #     cy = int(moment['m01']/moment['m00'])
        # except:
        #     return
        self.pub_image.publish(self.bridge.cv2_to_imgmsg(mask,"mono8"))

        # self.kalman.predict()
        # self.kalman.update(np.array([cx,cy,area]))

        # msg = Vector3()
        # msg.x = self.kalman.x[0]
        # msg.y = self.kalman.x[1]
        # msg.z = self.kalman.x[2]
        # self.pub_state.publish(msg)

        # print('Time:', rospy.get_time() - self.timestamp)


def ros_main(args = None) -> None:
    rospy.init_node('ev_seg')
    # RATE = 30
    seg_ros_class = seg_ros()
    # try:
    while not rospy.is_shutdown():
        rospy.spin()
    # except (KeyboardInterrupt, RuntimeError):
        # print('Shutting down seg_ros_class')
    # finally:
        # seg_ros_class.destroy_node()
    

if __name__ == "__main__":
    ros_main()
