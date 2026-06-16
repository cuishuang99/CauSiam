from PIL import Image
from glob import glob
import torchvision.transforms as transforms
import  os.path as osp
import random
import torch as T
import torchvision.transforms.functional as F
import torch.nn.functional as f
import cv2

class Mydataset(T.utils.data.Dataset):
    def __init__(self, phase,input_dir,gt_dir,cropH=None,cropW=None):
        self.cropH = cropH
        self.cropW = cropW
        self.phase = phase
        self.inputs = sorted(glob(osp.join(input_dir,"*")))
        self.gts = sorted(glob(osp.join(gt_dir,"*")))
        self.names = [osp.splitext(osp.basename(name))[0] for name in self.inputs]

        self.count = len(self.inputs)

        transform_list = []
        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def load_images_transform(self, file):
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = self.transform(img)
        return img
    
    def crop(self,input,gt):
        _,H,W = input.shape
        h = self.cropH
        w = self.cropW
        rnd_h = random.randint(0, max(0, H - h))
        rnd_w = random.randint(0, max(0, W - w))
        input = input[:, rnd_h:rnd_h + h, rnd_w:rnd_w + w]
        gt = gt[:, rnd_h:rnd_h + h, rnd_w:rnd_w + w]
        return input,gt
    
    def horizontalflip(self,input,gt):
        if random.random() < 0.5:
            return F.hflip(input),F.hflip(gt)
        return input,gt
    
    def pad(self,img,factor):
        h, w = img.shape[1], img.shape[2]
        H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        img = f.pad(img, (0, padw, 0, padh), 'reflect')
        return img
    
    def __getitem__(self, index):
        input = self.load_images_transform(self.inputs[index])
        gt = self.load_images_transform(self.gts[index])
        name = self.names[index]
        if self.phase == 'train':
            if self.cropH != None and self.cropW != None:
                input, gt = self.crop(input, gt)
            input, gt = self.horizontalflip(input, gt)
        else:
            # Pad so spatial size is divisible by 16 for downsampling
            input = self.pad(input, 16)
            gt = self.pad(gt, 16)
        
        return {'input':input,
                'gt':gt,
                'name':name}

    def __len__(self):
        return self.count
    
if __name__ == '__main__':
    train_dataset = Mydataset('train','/home/csh/dataset/DPDD/train/inputC',
                                '/home/csh/dataset/DPDD/train/target',
                                256,256)
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(train_dataset,batch_size=1,num_workers=0,pin_memory=True)
    data = iter(train_dataloader).next()
    input = data['input']
    gt = data['gt']
    import sys
    sys.path.append("..")
    from utils.img_utils import *
    save_tensor2img(input[0],'../input.png')
    save_tensor2img(gt[0],'../gt.png')
