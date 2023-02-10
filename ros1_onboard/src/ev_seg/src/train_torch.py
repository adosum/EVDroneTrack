#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 16:38:13 2021

@author: user
"""
from tqdm import tqdm
import torch
from unet_torch import UNet
from load_dataset import BasicDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from eval_torch import eval_net
import logging
from losses_torch import compute_photometric_loss,compute_event_warp_loss,smooth_loss,cross_entropy_two_layer
# from vis_flow import flow_viz_np
import os
import shutil
import torchvision.transforms as transforms
import datetime
import argparse

from skimage.measure import label, regionprops

parser = argparse.ArgumentParser(description='Spike-FlowNet Training on several datasets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', type=str, metavar='DIR', default='/home/user/Documents',
                    help='path to dataset')

parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--bias-decay', default=0, type=float,
                    metavar='B', help='bias decay')

args = parser.parse_args()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)
def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth.tar'))
        
def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

# dir_img = '/home/user/Documents/train_seq_room1_obj1/images/'
# dir_mask = '/home/user/Documents/train_seq_room1_obj1/masks/masks_full/'
# dir_event = '/home/user/Documents/train_seq_room1_obj1/events/'
image_size = 256
epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet(n_channels=3, n_classes=1, bilinear=True)
# net = torch.nn.DataParallel(net).cuda()
net.to(device=device)
if net.n_classes > 1:
    criterion = torch.nn.CrossEntropyLoss()
else:
    criterion = torch.nn.BCEWithLogitsLoss()

def train(train_loader, net, optimizer, epoch, train_writer):
    net.train()

    losses = AverageMeter()

    for batch in tqdm(train_loader):
        # imgs = batch['image']
        true_masks = batch['mask']
        eventframe = batch['eventframe']
        # next_eventframe = batch['next_eventframe']
        # next_img = batch['next_image']
        # imgs_e = torch.cat((imgs, next_img), 1)
        assert eventframe.shape[1] == net.n_channels, \
            f'Network has been defined with {net.n_channels} input channels, ' \
            f'but loaded images have {eventframe.shape[1]} channels. Please check that ' \
            'the images are loaded correctly.'

        # imgs = imgs.to(device=device, dtype=torch.float32)
        # imgs_e = imgs_e.to(device=device, dtype=torch.float32)
        # mask_type = torch.float32 if net.n_classes == 1 else torch.long
        
        true_masks = true_masks.to(device=device, dtype=torch.float32)
        # next_img = next_img.to(device=device, dtype=torch.float32)
        eventframe = eventframe.to(device=device, dtype=torch.float32)
        # next_eventframe = next_eventframe.to(device=device, dtype=torch.float32)
        # log_var_a = log_var_a.to(device=device, dtype=torch.float32)
        # log_var_b = log_var_b.to(device=device, dtype=torch.float32)
        pred, loss1,loss2,loss3 = net(eventframe)
        loss = criterion(pred, true_masks)
        
        # loss = compute_photometric_loss(imgs, 
        #                                 next_img,
        #                                 eventframe,
        #                                 pred,
        #                                 true_masks,
        #                                 [log_var_a, log_var_b])
        # pred = [pred,loss1,loss2,loss3]
        # loss = compute_photometric_loss(imgs, 
        #                                 next_img,
        #                                 eventframe,
        #                                 next_eventframe,
        #                                 pred,
        #                                 true_masks,
        #                                 [log_var_a, log_var_b])
        
        # loss = compute_event_warp_loss(eventframe,pred)
        # loss = cross_entropy_two_layer(true_masks,pred)
        # loss = loss + 10 * smooth_loss(pred) #+ cross_entropy_two_layer(true_masks,pred)
        

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(net.parameters(), 0.1)
        optimizer.step()
        losses.update(loss.item(), eventframe.size(0))
    
    print('Epoch: [{0}]\t Loss {1}'.format(epoch,losses))        
    return losses.avg

def validate(test_loader, net, epoch):
    net.eval()

    loss_sum = []

    for batch in test_loader:
        # imgs = batch['image']
        true_masks = batch['mask'].squeeze(1)
        masks = true_masks.squeeze(0).numpy()
        eventframe = batch['eventframe']
        # next_eventframe = batch['next_eventframe']
        # next_img = batch['next_image']
        # imgs_e = torch.cat((imgs, next_img), 1)

        # imgs = imgs.to(device=device, dtype=torch.float32)
        # imgs_e = imgs_e.to(device=device, dtype=torch.float32)
        # mask_type = torch.float32 if net.n_classes == 1 else torch.long
        
        true_masks = true_masks.to(device=device, dtype=torch.float32)
        # next_img = next_img.to(device=device, dtype=torch.float32)
        eventframe = eventframe.to(device=device, dtype=torch.float32)
        # next_eventframe = next_eventframe.to(device=device, dtype=torch.float32)
        # log_var_a = log_var_a.to(device=device, dtype=torch.float32)
        # log_var_b = log_var_b.to(device=device, dtype=torch.float32)
        pred = net(eventframe)
        # pred = torch.argmax(pred,1)
        pred = pred.squeeze(0).cpu().detach().permute(1,2,0).numpy()
        lbl_0 = label(pred) 
        props_pred = regionprops(lbl_0)
        
        lbl_0 = label(masks) 
        props = regionprops(lbl_0)
        count = 0
        for prop in props:
            boxA = prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2]
            for prop_pred in props_pred:
                boxB = prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2]
                iou = bb_intersection_over_union(boxA,boxB)
                if iou > 0.5:
                    count += 1
                    break
        if len(props)>0:    
            loss_sum.append(count/len(props))
        else:
            loss_sum.append(1)
    print('-------------------------------------------------------')
    print('Mean Det rate: {:.2f}, sum Det rate: {:.2f}'
                  .format(sum(loss_sum)/len(loss_sum), sum(loss_sum)))
    print('-------------------------------------------------------')
    return sum(loss_sum)/len(loss_sum)
    
    
def main():   

    
    co_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(30),
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
    ])
    
    # dir_img = '/opt/data/sun/data/train_seq_room3_obj3/images/'
    # dir_mask = '/opt/data/sun/data/train_seq_room3_obj3/masks_33/'
    # dir_event = '/opt/data/sun/data/train_seq_room3_obj3/events/'
    # image_size = 256
    # dataset1 = BasicDataset(dir_img, dir_mask,dir_event,image_size,co_transform)
    # dir_img = '/opt/data/sun/data/train_seq_room3_obj2/images/'
    # dir_mask = '/opt/data/sun/data/train_seq_room3_obj2/masks_32/'
    # dir_event = '/opt/data/sun/data/train_seq_room3_obj2/events/'
    # dataset2 = BasicDataset(dir_img, dir_mask,dir_event,image_size,co_transform)
    dir_img = os.path.join(os.path.join(args.data,'train_seq_room1_obj1'),'images/')
    dir_mask = os.path.join(os.path.join(args.data,'train_seq_room1_obj1'),'masks/masks_full/')
    dir_event = os.path.join(os.path.join(args.data,'train_seq_room1_obj1'),'events/')
    dataset3 = BasicDataset(dir_img, dir_mask,dir_event,image_size,co_transform)
    all_datasets = []
    all_datasets.append(dataset3)
    # all_datasets.append(dataset1)
    # all_datasets.append(dataset2)
    final_dataset = torch.utils.data.ConcatDataset(all_datasets)
    
    dir_img = os.path.join(os.path.join(args.data,'test_seq'),'images/')
    dir_mask = os.path.join(os.path.join(args.data,'test_seq'),'masks/masks_full/')
    dir_event = os.path.join(os.path.join(args.data,'test_seq'),'events/')
    test_dataset = BasicDataset(dir_img, dir_mask,dir_event,image_size)
    test_loader = DataLoader(test_dataset, shuffle=False,batch_size=1,  num_workers=8, pin_memory=True)
    
    batch_size = 2
    lr = 0.00001
    
    save_path = 'b{},lr{}'.format(
                batch_size,
                lr)
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    save_path = os.path.join(timestamp,save_path)
    save_path = os.path.join('segement_res',save_path)
    print('\n => Everything will be saved to {}'.format(save_path))
    
    train_loader = DataLoader(final_dataset, shuffle=True,batch_size=batch_size,  num_workers=8, pin_memory=True)
    # log_var_a = torch.zeros((1,), requires_grad=True)
    # log_var_b = torch.zeros((1,), requires_grad=True)
    param_groups = [{'params': net.bias_parameters(), 'weight_decay': args.bias_decay},
                    {'params': net.weight_parameters(), 'weight_decay': args.weight_decay}]
    
    
    optimizer = torch.optim.Adam(param_groups, lr=lr, weight_decay=1e-8)
    train_writer = SummaryWriter(os.path.join(save_path,'train'))
    test_writer = SummaryWriter(os.path.join(save_path,'test'))
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)

    best_EPE = -1
    for epoch in range(epochs):
        
        print('=> training started')
        train_loss = train(train_loader, net, optimizer, epoch, train_writer)
        train_writer.add_scalar('mean loss', train_loss, epoch)
     
        if (epoch + 1)%2 == 0:
            with torch.no_grad():
                EPE = validate(test_loader, net, epoch)
            test_writer.add_scalar('mean EPE', EPE, epoch)
            if best_EPE < 0:
                best_EPE = EPE
                
            test_writer.add_scalar('mean EPE', EPE, epoch)
            is_best = EPE < best_EPE
            best_EPE = min(EPE, best_EPE)
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),
                    'best_EPE': best_EPE,
                }, is_best, save_path)
            
if __name__ == '__main__':
    main()
                   
                        
                        

    
