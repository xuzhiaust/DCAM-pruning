from pytorchyolo.model_attention import load_model_attention,record_list
import torch
import cv2
import torchvision.transforms as transforms
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS
import numpy as np

yolo_attention=load_model_attention("G:\pytorchyolov3\pytorchyolo3_2\PyTorch-YOLOv3-master\config\yolov3.cfg")

img=cv2.imread('1.jpg')
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img=img.astype(np.float32)

img=torch.from_numpy(img).permute(2,0,1).unsqueeze(0).cuda()

# img2=torch.rand(1,3,416,416).cuda()
yolo_attention.eval()
out=yolo_attention(img)
