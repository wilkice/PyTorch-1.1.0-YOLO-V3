import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

def arg_parse():
    """
    parse arguments to the detect module
    """
    parser = argparse.ArgumentParser(description='YOLO-V3 object module')
    parser.add_argument(
        "--imagefolder", help='Directory containing imgs', default='imgs', type=str)
    parser.add_argument('--det', help='where to save imgs',
                        default='det', type=str)
    parser.add_argument('--bs', help='batch_size', default=1)
    parser.add_argument(
        '--confidence', help='whether to preserve the box', default=0.5)
    parser.add_argument('--nms_thre', help='nms threshold', default=0.4)
    parser.add_argument('--cfg', help='path to cfg file',
                        default='../yolov3.cfg')
    parser.add_argument('--weights', help='path to weights',
                        default='../yolov3.weights')
    parser.add_argument(
        '--reso', help='Input resolution of the network. Bigger to increase accuracy but decrease speed')
    return parser.parse_args()


arg = arg_parse()
img_folder = arg.imagefolder
batch_size = arg.bs
confidence = float(arg.confidence)
nms_thre = arg.nms_thre
start = 0


def load_classes(namesfile):
    with open(namesfile, 'r') as f:
        names = f.read().split('\n')[:-1]
        return names


num_classes = 80
classes = load_classes('../coco.names')


print('Loading network......')
model = Darknet(arg.cfg)
model.load_weights(arg.weights)
print('Network loaded successfully...')

#TODO: use resolution to speed
#TODO: CUDA SUPPORT

inp_dim = int(model.net_info['height'])

assert inp_dim % 32 ==0
assert inp_dim> 32

model.eval()

transform = transforms.Compose([
    transforms.Resize((608, 608)),
    transforms.ToTensor(),
])
dataset = dset.ImageFolder('imgs', transform=transform)
dataloader = DataLoader(dataset, batch_size=2)
img_name = os.listdir('imgs/a')
img_num = 0
for i in dataloader:
    for j in range(len(i)):
        try:
            with torch.no_grad():
                prediction = model(i[0][j].unsqueeze(0), CUDA=False)   
        except:
            print('all imgs has been checked')
        else:
            prediction = write_results(prediction, confidence, 80, 0.4)
                
            objs = [classes[int(x[-1])] for x in prediction]
            
            print(objs)
            
            box1 = np.array(prediction[0,1:5] * 225 / 608)
            box2 = np.array(prediction[1,1:5] * 225 / 608)
            dataset_notensor = dset.ImageFolder('imgs')
            im = dataset_notensor[img_num][0]
            print(box1)
            draw = ImageDraw.Draw(im)
            draw.rectangle(box1, outline='red', width=2)
            draw.rectangle(box2, outline='red', width=2)
            im.save('haha.png')
            img_num +=1
            

def draw_box(prediction):
    img = np.transpose(img,(1,2,0))
    print(img.shape)




# TODO: letterbox needed?


def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
                cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img


# img3 = write(prediction, img)
# cv2.imwrite('didi.png, img3')
