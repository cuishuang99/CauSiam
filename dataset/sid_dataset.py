from PIL import Image
from glob import glob
import torchvision.transforms as transforms
import  os.path as osp
import random
import torch as T
import torchvision.transforms.functional as F
import torch.nn.functional as f

class SID_dataset(T.utils.data.Dataset):
    def __init__(self, phase,input_dir,gt_dir,cropH=None,cropW=None):
        self.cropH = cropH
        self.cropW = cropW
        self.phase = phase
        
        self.data_info = {'path_LQ': [], 'path_GT': [],
                          'folder': [], 'idx': []}
        self.imgs_LQ, self.imgs_GT = {}, {}

        subfolders_LQ_origin = sorted(glob(osp.join(input_dir,"*")))
        subfolders_GT_origin = sorted(glob(osp.join(gt_dir,"*")))
        subfolders_LQ = []
        subfolders_GT = []
        if self.phase == 'train':
            for mm in range(len(subfolders_LQ_origin)):
                name = osp.basename(subfolders_LQ_origin[mm])
                if '0' in name[0] or '2' in name[0]:
                    subfolders_LQ.append(subfolders_LQ_origin[mm])
                    subfolders_GT.append(subfolders_GT_origin[mm])
        else:
            for mm in range(len(subfolders_LQ_origin)):
                name = osp.basename(subfolders_LQ_origin[mm])
                if '1' in name[0]:
                    subfolders_LQ.append(subfolders_LQ_origin[mm])
                    subfolders_GT.append(subfolders_GT_origin[mm])
        
        for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
            # for frames in each video:
            subfolder_name = osp.basename(subfolder_LQ)

            img_paths_LQ = sorted(glob(osp.join(subfolder_LQ,"*")))
            img_paths_GT = sorted(glob(osp.join(subfolder_GT,"*")))
            
            max_idx = len(img_paths_LQ)
            self.data_info['path_LQ'].extend(img_paths_LQ)  # list of path str of images
            self.data_info['path_GT'].extend(img_paths_GT)
            self.data_info['folder'].extend([subfolder_name] * max_idx)
            for i in range(max_idx):
                self.data_info['idx'].append('{}/{}'.format(i, max_idx))


            self.imgs_LQ[subfolder_name] = img_paths_LQ
            self.imgs_GT[subfolder_name] = img_paths_GT
                    
        transform_list = []
        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def load_images_transform(self, file):
        img = Image.open(file).convert('RGB')
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
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)

        img_LQ_path = self.imgs_LQ[folder][idx]
        img_GT_path = self.imgs_GT[folder][0]
        
        input = self.load_images_transform(img_LQ_path)
        gt = self.load_images_transform(img_GT_path)
        if self.phase == 'train':
            if self.cropH != None and self.cropW != None:
                input, gt = self.crop(input, gt)
            input, gt = self.horizontalflip(input, gt)
        else:
            # Pad so spatial size is divisible by 4 for downsampling
            input = self.pad(input, 4)
            gt = self.pad(gt, 4)
        
        return {'input':input,
                'gt':gt,
                'name':osp.splitext(osp.basename(img_LQ_path))[0]}

    def __len__(self):
        return len(self.data_info['path_LQ'])
    
if __name__ == '__main__':
    train_dataset = SID_dataset('train','/home/wjl/LLIE-datasets/sid_processed/short_sid2',
                                '/home/wjl/LLIE-datasets/sid_processed/long_sid2',
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
    
