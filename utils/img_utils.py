from typing import Any
import torch
import torch as T
import os
import os.path as osp
import cv2
from glob import glob
import numpy as np
import torchvision.transforms as transformers
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as f

metrics=['psnr', 'ssim', 'lpips']

class Lpips(float):
    def __gt__(self, other):
        return self < other

class Index():
    def __init__(self) -> None:
        self.metrics = metrics
        self.indexs = ()
        self.best_indexs = [0]*len(metrics)
        # self.sum_indexs = [0]*len(metrics)
        self.best_flags = [True]*len(metrics)
        # self.cnt = 0
    
    def __call__(self, *indexs) -> Any:
        # self.cnt += 1
        self.indexs = indexs
        # self.sum_indexs = [x+y for x,y in zip(self.indexs,self.sum_indexs)]
        self.best_indexs = [max(x,y) for x,y in zip(self.indexs,self.best_indexs)]
        self.best_flags = [x>y for x,y in zip(self.indexs,self.best_indexs)]
        
class Timer():
    def __init__(self) -> None:
        self.du = self.datetime_init()
        self.tot_du = self.datetime_init()
        
    def datetime_init(self):
        tmp = datetime.now()
        return tmp-tmp
    
    def __call__(self,du):
        self.du = du
        self.tot_du += self.du

def amp_pha2heatmap(img):
    img = tensornorm(img) / 2 + 0.5
    img = tensor2numpy(img)
    img = np.uint8(img * 255)
    img = cv2.applyColorMap(img,cv2.COLORMAP_WINTER)
    img = img[:,:,::-1]
    img = img / 255
    img = numpy2tensor(img)
    return img

def img2amp_pha_heatmap(img):
    fft = T.fft.fft2(img)
    fft = T.fft.fftshift(fft)
    amp = T.abs(fft)
    pha = T.angle(fft)
    amp = T.log(1+amp)
    
    amp = amp_pha2heatmap(amp)
    pha = amp_pha2heatmap(pha)
    return amp,pha

def tensor2numpyimg(img):
    img = T.clamp(img,0,1)
    img = tensor2numpy(img)
    img = (img * 255).astype(np.uint8)
    return img

def save_tensor2img(img, filename):
    img = tensor2numpyimg(img)
    img = Image.fromarray(img)
    img.save(filename)

def dir2imgs(dir):
    return sorted(glob(osp.join(dir,'*')))

def save_com_img(save_path,*img):
    save_dir = osp.dirname(osp.abspath(save_path))
    create_dir(save_dir)
    img_list = list(img)
    save_tensor2img(T.concat(img_list,dim=2),save_path)

def save_com_paths(save_path,*paths):
    save_dir = osp.dirname(osp.abspath(save_path))
    create_dir(save_dir)
    path_list = list(paths)
    path_list = [path2tensorimg(path) for path in path_list]
    save_com_img(save_path,*path_list)

def save_com_dirs(save_dir,*dirs):
    create_dir(save_dir)
    dir_list = list(dirs)
    dir_list = [dir2imgs(dir) for dir in dirs]
    dir_list = [save_com_paths(osp.join(save_dir,osp.basename(imgs[0])),*imgs) for imgs in tqdm(list(zip(*dir_list)))]

def create_dir(path):
    if not osp.exists(path):
        os.makedirs(path, exist_ok=True)

def path2tensorimg(path):
    trans = transformers.ToTensor()
    img = Image.open(path).convert('RGB')
    return trans(img)

def tensor2Y(img):
    trans2pil = transformers.ToPILImage()
    trans2ten = transformers.ToTensor()
    img = trans2pil(img).convert('YCbCr')
    img = trans2ten(img)
    return img[0:1, :, :]

def tensor2numpy(x):
    return x.detach().cpu().numpy().transpose(1,2,0)

def cam(x):
    x = np.uint8(255 * x)
    x = cv2.applyColorMap(x, cv2.COLORMAP_JET)
    x = x[:,:,::-1]
    return x / 255.0

def tensornorm(img):
    res = []
    for i in img:
        minx = torch.min(i)
        maxx = torch.max(i)
        i = (i-minx)/(maxx-minx)
        res.append(i)
    return T.stack(res)

def numpy2tensor(x):
    return torch.from_numpy(x).permute(2,0,1).float()

# Single-channel conversion
def tensor2heatmap(x):
    x = tensor2numpy(x)
    x = cam(x)
    x = numpy2tensor(x)
    return x

def tensor2gray(img):
    trans = transformers.Grayscale(1)
    return trans(img)

def pad(img,factor):
    h, w = img.shape[1], img.shape[2]
    H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
    padh = H-h if h%factor!=0 else 0
    padw = W-w if w%factor!=0 else 0
    img = f.pad(img, (0, padw, 0, padh), 'reflect')
    return img

if __name__ == '__main__':
    pass
