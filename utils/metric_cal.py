import torch
import os.path as osp
from glob import glob
from tqdm import tqdm
from .img_utils import *
from .metric_utils import *
import torchvision.transforms as transforms

metrics=['psnr', 'ssim', 'lpips']

def calculate_psnr(img1,
                   img2,
                   crop_border=0,
                   input_order='HWC',
                   test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        img2 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    if type(img1) == torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1,2,0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1,2,0)
        
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    max_value = 1. if img1.max() <= 1 else 255.
    return 20. * np.log10(max_value / np.sqrt(mse))

def calculate_ssim(img1,
                   img2,
                   crop_border=0,
                   input_order='HWC',
                   test_y_channel=False):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')

    if type(img1) == torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1,2,0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1,2,0)

    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
        return ssim_cly(img1[..., 0], img2[..., 0])


    ssims = []
    # ssims_before = []

    # skimage_before = skimage.metrics.structural_similarity(img1, img2, data_range=255., multichannel=True)
    # print('.._skimage',
    #       skimage.metrics.structural_similarity(img1, img2, data_range=255., multichannel=True))
    max_value = 1 if img1.max() <= 1 else 255
    with torch.no_grad():
        final_ssim = ssim_3d(img1, img2, max_value)
        ssims.append(final_ssim)

    # for i in range(img1.shape[2]):
    #     ssims_before.append(_ssim(img1, img2))

    # print('..ssim mean , new {:.4f}  and before {:.4f} .... skimage before {:.4f}'.format(np.array(ssims).mean(), np.array(ssims_before).mean(), skimage_before))
        # ssims.append(skimage.metrics.structural_similarity(img1[..., i], img2[..., i], multichannel=False))

    return np.array(ssims).mean()

# tensor,[0,1]
def calculate_lpips(out_imgs,gt_imgs,test_y_channel=False):
    import lpips
    if test_y_channel:
        # [0,1]->[-1,1]
        trans = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        out_imgs = [trans(x) for x in out_imgs]
        gt_imgs = [trans(x) for x in gt_imgs]
        
    out_imgs = T.stack(out_imgs,dim=0)
    gt_imgs = T.stack(gt_imgs,dim=0)
    
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    res =  loss_fn_alex(out_imgs.cuda(),gt_imgs.cuda()).detach().float().cpu()
    del loss_fn_alex
    
    return res

def cal_index_imgs(out_imgs,gt_imgs,y_channel,tqdm_flag = False):
    score_psnr_all = []
    iters = enumerate(tqdm(list(zip(out_imgs,gt_imgs)),disable=not tqdm_flag))
    for i, img in iters:
        img_out,img_gt = img
        img_out_ = tensor2numpyimg(img_out)
        img_gt_ = tensor2numpyimg(img_gt)
        score_psnr_all.append(calculate_psnr(img_out_,img_gt_,test_y_channel=y_channel))
    
    return sum(score_psnr_all)/len(score_psnr_all)

def cal_index_dir(result_path, gt_path, y_channel,tqdm_flag = False):
    score_psnr_all = []
    
    img_out_paths = sorted(glob(osp.join(result_path, "*.*")))
    img_gt_paths = sorted(glob(osp.join(gt_path,"*.*")))
    
    iters = enumerate(tqdm(list(zip(img_out_paths,img_gt_paths)),disable=not tqdm_flag))
    for i, img_path in iters:
        img_out_path,img_gt_path = img_path
        img_out_ = cv2.imread(img_out_path,cv2.IMREAD_UNCHANGED)
        img_gt_ = cv2.imread(img_gt_path,cv2.IMREAD_UNCHANGED)
        score_psnr_all.append(calculate_psnr(img_out_,img_gt_,test_y_channel=y_channel))

    return sum(score_psnr_all)/len(score_psnr_all)

# def cal_index_imgs(out_imgs,gt_imgs,y_channel,tqdm_flag = False):
#     # Import pyiqa inside the function because it initializes CUDA;
#     # CUDA_VISIBLE_DEVICES must be set before importing pyiqa.
#     import pyiqa

#     # Initialize metrics
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     iqa_psnr, iqa_ssim, iqa_lpips = None, None, None
#     score_psnr_all, score_ssim_all, score_lpips_all = [], [], []
#     if 'psnr' in metrics:
#         iqa_psnr = pyiqa.create_metric('psnr').to(device)
#         iqa_psnr.eval()
#     if 'ssim' in metrics:
#         iqa_ssim = pyiqa.create_metric('ssim').to(device)
#         iqa_ssim.eval()
#     if 'lpips' in metrics:
#         # iqa_lpips = pyiqa.create_metric('lpips').to(device)
#         iqa_lpips = pyiqa.create_metric('lpips-vgg').to(device)
#         iqa_lpips.eval()
        
#     iters = enumerate(tqdm(list(zip(out_imgs,gt_imgs)),disable=tqdm_flag))
#     for i, img in iters:
#         img_out,img_gt = img
#         if y_channel:
#             img_out = tensor2Y(img_out)
#             img_gt = tensor2Y(img_gt)
#         with torch.no_grad():
#             img_out = img_out.unsqueeze(0).to(device)
#             img_gt = img_gt.unsqueeze(0).to(device)
#             if iqa_psnr is not None:
#                 score_psnr_all.append(iqa_psnr(img_out, img_gt).item())
#             if iqa_ssim is not None:
#                 score_ssim_all.append(iqa_ssim(img_out, img_gt).item())
#             if iqa_lpips is not None:
#                 score_lpips_all.append(iqa_lpips(img_out, img_gt).item())
#     return sum(score_psnr_all)/len(score_psnr_all), sum(score_ssim_all)/len(score_ssim_all), sum(score_lpips_all)/len(score_lpips_all)

# def cal_index_dir(result_path, gt_path, y_channel,tqdm_flag = False):
#     # Import pyiqa inside the function because it initializes CUDA;
#     # CUDA_VISIBLE_DEVICES must be set before importing pyiqa.
#     import pyiqa

#     # Initialize metrics
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     iqa_psnr, iqa_ssim, iqa_lpips = None, None, None
#     score_psnr_all, score_ssim_all, score_lpips_all = [], [], []
#     if 'psnr' in metrics:
#         iqa_psnr = pyiqa.create_metric('psnr').to(device)
#         iqa_psnr.eval()
#     if 'ssim' in metrics:
#         iqa_ssim = pyiqa.create_metric('ssim').to(device)
#         iqa_ssim.eval()
#     if 'lpips' in metrics:
#         # iqa_lpips = pyiqa.create_metric('lpips').to(device)
#         iqa_lpips = pyiqa.create_metric('lpips-vgg').to(device)
#         iqa_lpips.eval()

#     img_out_paths = sorted(glob(osp.join(result_path, "*.*")))
#     img_gt_paths = sorted(glob(osp.join(gt_path,"*.*")))
    
#     iters = enumerate(tqdm(list(zip(img_out_paths,img_gt_paths)),disable=tqdm_flag))
#     for i, img_path in iters:
#         img_out_path,img_gt_path = img_path
        
#         # img_out = cv2.imread(
#         #     img_out_path, cv2.IMREAD_UNCHANGED).astype(np.float32)/255.
#         # img_out = np.transpose(img_out, (2, 0, 1))
#         # img_out = torch.from_numpy(img_out).float()
#         img_out = path2tensorimg(img_out_path)
#         if y_channel:
#             img_out = tensor2Y(img_out)

#         # img_gt = cv2.imread(
#         #     img_gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32)/255.
#         # img_gt = np.transpose(img_gt, (2, 0, 1))
#         # img_gt = torch.from_numpy(img_gt).float()
#         img_gt = path2tensorimg(img_gt_path)
#         if y_channel:
#             img_gt = tensor2Y(img_gt)
#         with torch.no_grad():
#             img_out = img_out.unsqueeze(0).to(device)
#             img_gt = img_gt.unsqueeze(0).to(device)
#             if iqa_psnr is not None:
#                 score_psnr_all.append(iqa_psnr(img_out, img_gt).item())
#             if iqa_ssim is not None:
#                 score_ssim_all.append(iqa_ssim(img_out, img_gt).item())
#             if iqa_lpips is not None:
#                 score_lpips_all.append(iqa_lpips(img_out, img_gt).item())
#     return sum(score_psnr_all)/len(score_psnr_all), sum(score_ssim_all)/len(score_ssim_all), sum(score_lpips_all)/len(score_lpips_all)

if __name__ == '__main__':
    pass
