# imageDenoiseUsingUNet-VGG
通用图像去噪网络，以UNet为基本框架，VGG16的卷积层作为编码器


## 环境：
Python 3.5.3 <br>
Pytorch 1.2.0 <br>


## 使用方法：
1、准备成对的数据集，以相同的名字分别放置在data/src和data/target文件夹下 <br>
2、如果图像的尺寸过大&&电脑内存不充足，可利用data/文件夹下的脚本进行剪切处理 <br>
3、使用train.py进行训练 <br>


## 实验结果：
数据集：甲骨拓片数据集。 <br>
对比网络：CycleGAN <br>
UNet-VGG的效果优于CycleGAN，可以看出CycleGAN对于字体有弱化现象，且对于细节的处理没有UNet-VGG好。 <br>
### CycleGAN处理效果
![image](https://github.com/libai-github/imageDenoiseUsingUNet-VGG/blob/master/resultOfCycleGAN.png)
### UNet-VGG处理效果
![image](https://github.com/libai-github/imageDenoiseUsingUNet-VGG/blob/master/resultOfUNet-VGG.png)

