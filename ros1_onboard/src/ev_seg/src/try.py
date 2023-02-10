from Models import U_Net
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

img = 'event_00000561.png'

model = U_Net(3,2)
ckpt_file = '/home/haixin/ros1/src/ev_seg/src/model_best.pth.tar'
ckpt = torch.load(ckpt_file)
model.load_state_dict(ckpt["state_dict"])
model = model.cuda()
model.eval()

img_rgb = Image.open(img)
img_rgb = img_rgb.resize((256, 256))
img_rgb = np.array(img_rgb)
img_rgb = img_rgb/255
img_rgb = img_rgb.transpose((2, 0, 1))
img_rgb = torch.tensor(img_rgb).unsqueeze(0).float().cuda()
mask = model(img_rgb)

mask = torch.argmax(mask,1)
mask = mask.squeeze(0).detach().cpu().numpy().astype(np.uint8)

plt.imshow(mask.astype(float))
plt.show()
