import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from net import VGG16UNet
from loss import My_loss

from utils.dataset import MyDataset
from torch.utils.data import DataLoader

from torchsummary import summary

np.set_printoptions(threshold = np.inf) 
np.set_printoptions(suppress = True)
from PIL import Image  


dir_src = 'data/src/'
dir_target = 'data/target/'
dir_checkpoint = 'checkpoints/'


def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              save_cp=True,
              img_scale=0.5,
              out_channel=3):
    #数据加载器
    dataset = MyDataset(dir_src, dir_target, out_channel, img_scale)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    #优化器损失函数
    optimizer = optim.Adam(net.parameters(),lr=lr)
    criterion = torch.nn.MSELoss()

    
    for epoch in range(101,epochs):
        #调整学习率
        if epoch % 20 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.5


        net.train()
        epoch_loss = 0
        global_step=0
        for batch in train_loader:
            #获取数据并加载到CPU/GPU
            src = batch['src']
            target = batch['target']
            src = src.to(device=device, dtype=torch.float32)
            target = target.to(device=device, dtype=torch.float32)  

            optimizer.zero_grad()   
            src_pred = net(src)#前向传播结果
            '''
            dis=torch.tensor(src_pred).cpu().numpy()
            print('src_pred:',np.sum(dis))
            print(dis[0][0][0])
            #print(dis[0][0][0])
            
            
            if global_step==5:
                image_src=(torch.tensor(src).cpu().numpy()[0][0])*255
                image_target=(torch.tensor(target).cpu().numpy()[0][0])*255
                img_src=Image.fromarray(image_src)
                img_src.show()
                img_target=Image.fromarray(image_target)
                img_target.show()
            '''
            
            
            loss = criterion(src_pred, target)#计算损失
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            if global_step%100==0:
                print(src_pred[0][0][0][0])
                print('global_step:',global_step,' ; loss:',loss.item())
            global_step+=1
            
        print('-------------------------------------------epoch:',epoch,' ; loss_mean:',epoch_loss/len(train_loader))
 
        #保存模型     
        if save_cp and epoch%10==0:
            try:
                os.mkdir(dir_checkpoint)
                print('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'Checkpoint_epoch{epoch}.pth')
            print(f'Checkpoint {epoch} saved !')


def get_args():
    parser = argparse.ArgumentParser(description='Train the VGG16UNet on srcimages and target',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=500,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=True,
                        help='Load model from a .pth file')
    parser.add_argument('-p', '--pretrained',dest='pretrained', type=str, default=False,
                        help='Load pretrained model from internet')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-o', '--out_channel', dest='out_channel', type=int, default=3,
                        help='channel of the images')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #可自动加载预训练模型的参数：将事先下载好的模型直接复制到缓存目录以节省临时下载耗费时间。
    net = VGG16UNet(out_channel=args.out_channel, pretrained=args.pretrained)
    
    #初始化Conv2d,ReLU,MaxPool2d,ConvTranspose2d,
    num=0
    for m in net.modules():
        if isinstance(m,nn.Conv2d) and num>14:#在此之前为VGG16参数，不用再初始化
            torch.nn.init.xavier_normal_(m.weight.data)#Xavier 正态分布
            if m.bias is not None:
                m.bias.data.zero_()
        elif  isinstance(m,nn.ReLU):
            continue
        elif isinstance(m,nn.MaxPool2d):
            continue
        elif isinstance(m,nn.ConvTranspose2d):
            torch.nn.init.xavier_normal_(m.weight.data)#Xavier 正态分布
            if m.bias is not None:
                m.bias.data.zero_()
        num+=1

    #加载上次训练的参数
    net.load_state_dict(torch.load('Checkpoint_epoch100l1.pth'))
    

    #加载数据到CPU/GPU
    net.to(device=device,dtype=torch.float32)
    #summary(net, (1, 224, 224)) 

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  out_channel=args.out_channel
                  )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
