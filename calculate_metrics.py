import os
import numpy as np
import torch
# from hdf5storage import loadmat
import scipy.io as sio
from torch import nn
from math import exp
from torch.autograd import Variable
import torch.nn.functional as F


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=255):
        Itrue = im_true.clamp(0., 1.)*data_range
        Ifake = im_fake.clamp(0., 1.)*data_range
        err = Itrue-Ifake
        err = torch.pow(err, 2)
        err = torch.mean(err, dim=0)
        err = torch.mean(err, dim=0)

        psnr = 10. * torch.log10((data_range ** 2) / err)
        psnr = torch.mean(psnr)
        return psnr


class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs.clamp(0., 1.)*255 - label.clamp(0., 1.)*255
        sqrt_error = torch.pow(error, 2)
        # view(-1)表示将张量展平为一维张量
        rmse = torch.sqrt(torch.mean(sqrt_error.contiguous().view(-1)))
        return rmse

class Loss_SAM(nn.Module):
    def __init__(self):
        super(Loss_SAM, self).__init__()
        # 防止除0错误
        self.eps = 2.2204e-16
    def forward(self,im1, im2):
        assert im1.shape == im2.shape
        H, W, C = im1.shape
        im1 = np.reshape(im1, (H*W, C))
        im2 = np.reshape(im2, (H*W, C))
        # 计算对应元素乘积(元素级乘法)
        core = np.multiply(im1, im2)
        mole = np.sum(core, axis=1)
        im1_norm = np.sqrt(np.sum(np.square(im1), axis=1))
        im2_norm = np.sqrt(np.sum(np.square(im2), axis=1))
        deno = np.multiply(im1_norm, im2_norm)
        # clip(-1, 1)将值限制在-1到1之间
        # np.rad2deg()将弧度转化为度数
        sam = np.rad2deg(np.arccos(((mole+self.eps)/(deno+self.eps)).clip(-1, 1)))
        return np.mean(sam)

class Loss_SAM_1(nn.Module):
    def __init__(self):
        super(Loss_SAM_1, self).__init__()
        # 防止除0错误
        self.eps = 2.2204e-16
    def forward(self,im1, im2):
        assert im1.shape == im2.shape
        H, W, C = im1.shape
        # im1 = np.reshape(im1, (H*W, C))
        im1 = im1.reshape(H * W, C)
        # im2 = np.reshape(im2, (H*W, C))
        im2 = im2.reshape(H * W, C)
        # 计算对应元素乘积(元素级乘法)
        core = im1 * im2
        mole = torch.sum(core, dim=1)
        # im1_norm = np.sqrt(np.sum(np.square(im1), axis=1))
        im1_norm = torch.sqrt(torch.sum(torch.square(im1), dim=1))
        # im2_norm = np.sqrt(np.sum(np.square(im2), axis=1))
        im2_norm = torch.sqrt(torch.sum(torch.square(im2), dim=1))
        deno = im1_norm * im2_norm
        # clip(-1, 1)将值限制在-1到1之间
        # np.rad2deg()将弧度转化为度数
        # sam = np.rad2deg(np.arccos(((mole+self.eps)/(deno+self.eps)).clip(-1, 1)))
        sam = torch.rad2deg(torch.acos(torch.clamp((mole + self.eps)/(deno + self.eps), -1, 1)))
        return torch.mean(sam)

class Loss_SSIM(nn.Module):
    def __init__(self):
        super(Loss_SSIM, self).__init__()
        pass

    def forward(self, img1, img2, window_size=11, size_average=True):
        (_, channel, _, _) = img1.size()
        window = create_window(window_size, channel)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)

        return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

# 输入四维张量
class Loss_ERGAS(nn.Module):
    def __init__(self):
        super(Loss_ERGAS, self).__init__()


    def forward(self, img_tgt, img_fus):
        scale = 8
        img_tgt = img_tgt.squeeze(0).data.cpu().numpy()
        img_fus = img_fus.squeeze(0).data.cpu().numpy()
        img_tgt = np.squeeze(img_tgt)
        img_fus = np.squeeze(img_fus)
        img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
        img_fus = img_fus.reshape(img_fus.shape[0], -1)

        rmse = np.mean((img_tgt - img_fus) ** 2, axis=1)
        rmse = rmse ** 0.5
        mean = np.mean(img_tgt, axis=1)

        ergas = np.mean((rmse / mean) ** 2)
        ergas = 100 / scale * ergas ** 0.5

        return ergas
    
class Loss_ERGAS_1(nn.Module):
    def __init__(self):
        super(Loss_ERGAS_1, self).__init__()


    def forward(self, img_tgt, img_fus):
        scale = 8
        img_tgt = img_tgt.squeeze(0)
        img_fus = img_fus.squeeze(0)
        img_tgt = img_tgt.squeeze()
        img_fus = img_fus.squeeze()
        img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
        img_fus = img_fus.reshape(img_fus.shape[0], -1)

        rmse = torch.mean((img_tgt - img_fus) ** 2, dim=1)
        rmse = rmse ** 0.5
        mean = torch.mean(img_tgt, dim=1)

        ergas = torch.mean((rmse / mean) ** 2)
        ergas = 100 / scale * ergas ** 0.5

        return ergas


if __name__ == '__main__':
    SAM = Loss_SAM()
    RMSE = Loss_RMSE()
    PSNR = Loss_PSNR()
    psnr_list = []
    sam_list = []
    sam = AverageMeter()
    rmse = AverageMeter()
    psnr = AverageMeter()
    path = 'D:\LYY\YJX_fusion\model_save\\fusion_model_v9_1\cavee_test/'
    imglist = os.listdir(path)

    for i in range(0, len(imglist)):
        img = sio.loadmat(path + imglist[i])
        lable = img["rea"]
        recon = img["fak"]
        sam_temp = SAM(lable, recon)
        psnr_temp = PSNR(torch.Tensor(lable), torch.Tensor(recon))
        sam.update(sam_temp)
        rmse.update(RMSE(torch.Tensor(lable), torch.Tensor(recon)))
        psnr.update(psnr_temp)
        psnr_list.append(psnr_temp)
        sam_list.append(sam_temp)
    print(sam.avg)
    print(rmse.avg)
    print(psnr.avg)
    print(psnr_list)
    print(sam_list)