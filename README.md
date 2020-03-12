# imageDenoiseUsingUNet-VGG
通用去噪网络，以UNet为基本框架，VGG16的卷积层作为编码器

环境：
Python 3.5.3
Pytorch 1.2.0

使用方法：


实验结果：
在甲骨拓片数据集上的效果优于CycleGAN,可以看出UNetVGG对于对于细节的处理更好。
![image](https://github.com/libai-github/imageDenoiseUsingUNet-VGG/blob/master/resultOfCycleGAN.png)

![image](https://github.com/libai-github/imageDenoiseUsingUNet-VGG/blob/master/resultOfUNet-VGG.png)

