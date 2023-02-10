#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 16:05:37 2021

@author: haixin
"""

import os
import time

import cv2
from numpy import empty

import torch
import numpy as np
from ev_segment.unet_torch import UNet

#from yolox.data.data_augment import preproc
#from yolox.data.datasets import COCO_CLASSES
#from yolox.exp import get_exp


import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import ParameterType

from std_msgs.msg import Header
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from skimage.measure import label, regionprops

#from bboxes_ex_msgs.msg import BoundingBoxes
#from bboxes_ex_msgs.msg import BoundingBox

# from darknet_ros_msgs.msg import BoundingBoxes
# from darknet_ros_msgs.msg import BoundingBox

#class Predictor(object):
#    def __init__(self, model, exp, cls_names=COCO_CLASSES, trt_file=None, decoder=None):
#        self.model = model
#        self.cls_names = cls_names
#        self.decoder = decoder
#        self.num_classes = exp.num_classes
#        self.confthre = exp.test_conf
#        self.nmsthre = exp.nmsthre
#        self.test_size = exp.test_size
#        if trt_file is not None:
#            from torch2trt import TRTModule
#            model_trt = TRTModule()
#            model_trt.load_state_dict(torch.load(trt_file))
#
#            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
#            self.model(x)
#            self.model = model_trt
#        self.rgb_means = (0.485, 0.456, 0.406)
#        self.std = (0.229, 0.224, 0.225)
#
#    def inference(self, img):
#        img_info = {'id': 0}
#        if isinstance(img, str):
#            img_info['file_name'] = os.path.basename(img)
#            img = cv2.imread(img)
#        else:
#            img_info['file_name'] = None
#
#        height, width = img.shape[:2]
#        img_info['height'] = height
#        img_info['width'] = width
#        img_info['raw_img'] = img
#
#        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
#        img_info['ratio'] = ratio
#        img = torch.from_numpy(img).unsqueeze(0).cuda()
#
#        with torch.no_grad():
#            outputs = self.model(img)
#            if self.decoder is not None:
#                outputs = self.decoder(outputs, dtype=outputs.type())
#        return outputs, img_info
#
#    def visual(self, output, img_info, cls_conf=0.35):
#        ratio = img_info['ratio']
#        img = img_info['raw_img']
#        if output is None:
#            return img
#        output = output.cpu()
#
#        bboxes = output[:, 0:4]
#
#        # preprocessing: resize
#        bboxes /= ratio
#
#        cls = output[:, 6]
#        scores = output[:, 4] * output[:, 5]
#
#        
#        return bboxes, scores, cls, self.cls_names

class seg_ros(Node):
    def __init__(self) -> None:

        # ROS2 init
        super().__init__('seg_ros')

        self.setting_seg_config()

        
        self.bridge = CvBridge()
        
#        self.pub = self.create_publisher(BoundingBoxes,"seg/bounding_boxes", 10)
        self.pub_image = self.create_publisher(Image,"seg/image_raw", 10)
        self.sub = self.create_subscription(Image,"event_frame",self.imageflow_callback, 10)

    def setting_seg_config(self) -> None:
        # set environment variables for distributed training
        
        # ==============================================================

        WEIGHTS_PATH = ''

        ckpt_file_description = ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description='CKPT File Path')

        self.declare_parameter('ckpt_file', WEIGHTS_PATH, ckpt_file_description)
#        self.declare_parameter('img_size', 640)
#        self.declare_param1eter('image_size/width', 640)
#        self.declare_parameter('image_size/height', 480)

        self.ckpt_file = self.get_parameter('ckpt_file').value

#        ckpt_file = self.get_parameter('ckpt_file').value
        #ckpt = torch.load(self.ckpt_file)
        self.model = UNet(n_channels=3,n_classes=2,bilinear=True)
        #self.model.load_state_dict(ckpt["model"])
        self.model = self.model.cuda() 
        self.model.eval()



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

    def imageflow_callback(self,msg:Image) -> None:
        img_rgb = self.bridge.imgmsg_to_cv2(msg,"bgr8")
        img_rgb = np.array(img_rgb)
        img_rgb = img_rgb.transpose((2, 0, 1))
        img_rgb = torch.tensor(img_rgb).unsqueeze(0).float().cuda()
        mask = self.model(img_rgb)
        mask = torch.argmax(mask,1)
        mask = mask.squeeze(0).detach().cpu().numpy().astype(np.uint8)
        mask = mask*255
        self.pub_image.publish(self.bridge.cv2_to_imgmsg(mask,"mono8"))
#            output = self.model(img_rgb)
            
#            try:
#                bboxes = self.mask_to_bbox(output)
                
#                bboxes = self.yolox2bboxes_msgs(bboxes, scores, cls, cls_names, msg.header)
#                self.pub.publish(bboxes)
#                self.pub_image.publish(self.bridge.cv2_to_imgmsg(img_rgb,"bgr8"))
#
#                if (self.imshow_isshow):
#                    cv2.imshow("YOLOX",result_img_rgb)
#                    cv2.waitKey(1)
#                
#            except:
#                if (self.imshow_isshow):
#                    cv2.imshow("YOLOX",img_rgb)
#                    cv2.waitKey(1)


def ros_main(args = None) -> None:
    rclpy.init(args=args)

    seg_ros_class = seg_ros()
    try:
        rclpy.spin(seg_ros_class)
    except (KeyboardInterrupt, RuntimeError):
        print('Shutting down seg_ros_class')
    finally:
        seg_ros_class.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == "__main__":
    ros_main()
