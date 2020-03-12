import os
from PIL import Image
import numpy as np
import math

if __name__ == "__main__":
    root='src'
    size=512
    for i in os.listdir(root):
        path=os.path.join(root,i)
        img=Image.open(path).convert('L')
        if img.size[0]<512:
            a,b=img.size
            img=img.resize((512,int(b*(512/a))))
            print('chuxian')
        if img.size[1]<512:
            a,b=img.size
            img=img.resize((int(a*(512/b)),512))
            print('chuxian')
        img=np.array(img)
        print(img.shape)
        
        h,w=img.shape
        y_max=math.ceil(h/size)
        x_max=math.ceil(w/size)
        for y in range(y_max):
            for x in range(x_max):
                if (y+1==y_max) and (x+1<x_max):
                    Image.fromarray(img[h-size:h,x*size:(x+1)*size]).save(path[0:len(path)-4]+'_'+str(h)+'_'+str(w)+'_'+str(y_max)+'_'+str(x_max)+'_'+str(y)+'_'+str(x)+'_.png')
                elif (x+1==x_max) and (y+1<y_max):
                    Image.fromarray(img[y*size:(y+1)*size,w-size:w]).save(path[0:len(path)-4]+'_'+str(h)+'_'+str(w)+'_'+str(y_max)+'_'+str(x_max)+'_'+str(y)+'_'+str(x)+'_.png')
                elif (y+1==y_max) and (x+1==x_max):
                    Image.fromarray(img[h-size:h,w-size:w]).save(path[0:len(path)-4]+'_'+str(h)+'_'+str(w)+'_'+str(y_max)+'_'+str(x_max)+'_'+str(y)+'_'+str(x)+'_.png')
                else:
                    Image.fromarray(img[y*size:(y+1)*size,x*size:(x+1)*size]).save(path[0:len(path)-4]+'_'+str(h)+'_'+str(w)+'_'+str(y_max)+'_'+str(x_max)+'_'+str(y)+'_'+str(x)+'_.png')
        os.remove(path)

        







