import os
from PIL import Image
import numpy as np
import shutil

if __name__ == "__main__":
    
    a='src'
    b='target'
    
    for i in os.listdir(a):
        imga=Image.open(os.path.join(a,i))
        imgb=Image.open(os.path.join(b,i))
        if imga.size!=imgb.size:
            print(imga.size,imgb.size)
            #imgb=imgb.resize(imga.size)
            #imgb.save(os.path.join(b,i))
            #os.remove(os.path.join(a,i))
            #os.remove(os.path.join(b,i))
            #shutil.move(os.path.join(a,i), os.path.join(c,i))
            #shutil.move(os.path.join(b,i), os.path.join(d,i))
            print(i)
    '''
    
    a='src'
    b='target'
    num=0
    for i in os.listdir(a):
        
        imga=Image.open(os.path.join(a,i))
        imgb=Image.open(os.path.join(b,i))
        if imga.size[0]<300 or imga.size[1]<300:
            print(imga.size,imgb.size)
            print(i)
            num+=1
    print(num)
        
    '''
    
    
        







