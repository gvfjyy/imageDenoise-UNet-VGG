import os
from PIL import Image

if __name__ == "__main__":        
    a='src'
    b='target'
    id=1000
    for i in os.listdir(a):
        img=Image.open(os.path.join(a,i))
        img.convert('L')
        os.remove(os.path.join(a,i))
        img.save(a+'/'+str(id)+'.png')
        
        img=Image.open(os.path.join(b,i))
        img.convert('L')
        os.remove(os.path.join(b,i))
        img.save(b+'/'+str(id)+'.png')
        
        
        print(id)
        id+=1
        
        
        
        
        





