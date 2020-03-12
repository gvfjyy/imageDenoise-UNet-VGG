from os.path import splitext
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
import os
import random
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, dir_src ,dir_target ,out_channel, scale):
        self.dir_src=dir_src
        self.dir_target=dir_target
        self.scale = scale
        self.out_channel=out_channel
        self.full=os.listdir(dir_src)
        self.img_src=[]
        self.img_target=[]
        for i in self.full:
            self.img_src.append(os.path.join(dir_src,i))
            self.img_target.append(os.path.join(dir_target,i))
            
    def __len__(self):
        return len(self.full)
    
    def __getitem__(self, i):
        src=Image.open(self.img_src[i]).convert('RGB')
        target=Image.open(self.img_target[i]).convert('RGB')
        if self.out_channel==1:
            src=src.convert('L')
            target=target.convert('L')
        src,target= self.preprocess(src, target, self.out_channel, self.scale)
        #转化为tensor
        return {'src': torch.from_numpy(src), 'target': torch.from_numpy(target)}
    
    def preprocess(cls, img1 ,img2 , out_channel, scale):
        w,h=img1.size
        if random.randint(0,2)==0:
            t=random.randint(0,359)
            img1=img1.rotate(t)#旋转
            img1=img1.crop((int(w*0.15), int(h*0.15), int(w*0.85), int(h*0.85)))#裁剪
            img1=img1.resize((w,h))#恢复原大小  
            img2=img2.rotate(t)#旋转
            img2=img2.crop((int(w*0.15), int(h*0.15), int(w*0.85), int(h*0.85)))#裁剪
            img2=img2.resize((w,h))#恢复原大小  

        t=random.randint(0,10)#翻转
        if t==0:
            img1=img1.transpose(Image.FLIP_LEFT_RIGHT)
            img2=img2.transpose(Image.FLIP_LEFT_RIGHT)
        elif t==1:
            img1=img1.transpose(Image.FLIP_TOP_BOTTOM)
            img2=img2.transpose(Image.FLIP_TOP_BOTTOM)
        elif t==2:
            img1=img1.transpose(Image.ROTATE_90)
            img2=img2.transpose(Image.ROTATE_90)
        elif t==3:
            img1=img1.transpose(Image.ROTATE_180)
            img2=img2.transpose(Image.ROTATE_180)
        elif t==4:
            img1=img1.transpose(Image.ROTATE_270)
            img2=img2.transpose(Image.ROTATE_270)
        elif t==5:
            img1=img1.transpose(Image.TRANSPOSE)
            img2=img2.transpose(Image.TRANSPOSE)
        elif t==6:
            img1=img1.transpose(Image.TRANSVERSE)
            img2=img2.transpose(Image.TRANSVERSE)
        
        img1=np.array(img1)
        img2=np.array(img2)
        if out_channel==3:
            # HWC to CHW
            img1 = np.array(img1).transpose((2, 0, 1))
            img2 = np.array(img2).transpose((2, 0, 1))
        elif out_channel==1:
            img1=img1.reshape(-1,img1.shape[0],img1.shape[1])
            img2=img2.reshape(-1,img2.shape[0],img2.shape[1])

        #归一化至[0,1]        
        img1 = (img1 / 255)
        img2 = (img2 / 255)
        return img1, img2
