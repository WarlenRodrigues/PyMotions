from PIL import Image
import glob
import numpy as np

class Throught_Img_Folder:
    def __init__(self,path):
        self.path=path
        self.image_list = []
        self.i=0
        for filename in glob.glob(path+'/*.jpg'):
            im=Image.open(filename)
            self.image_list.append(np.array(im))
    def read(self):
        print(self.i)
        return True, self.image_list[self.i]
        self.i+=1
    def isOpened(self):

        return True
        # if self.image_list :