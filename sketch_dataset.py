from preprocess import *

from chainer.dataset import dataset_mixin
import os

class SketchDataset(dataset_mixin.DatasetMixin):
    def __init__(self, dataDir='./dataset/sketch', size=384, train=True):
        print("load dataset start")
        if train:
            self.dataDir = os.path.join(dataDir,'train')
        else:
            self.dataDir = os.path.join(dataDir,'val')
        print("    from: %s"%self.dataDir)
        self.size = size
        self.dataset = loadimgs(self.dataDir, size)
    
    def __len__(self):
        return len(self.dataset)

    # return (img, label)
    def get_example(self, i):
        return self.dataset[i][0],self.dataset[i][1]

    def reloadimgs(self):
        self.dataset = loadimgs(self.dataDir, self.size)



    
