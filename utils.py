import torch
import os
os.environ['FOR_IGNORE_EXCEPTIONS'] = '1'
import glob
from scipy import signal
from numpy import *
from torch import nn
import numpy as np
import re

def make_dir(path):
    if not os.path.exists(path):
        # 如果目录不存在，则创建它
        os.makedirs(path)
        print(f"目录 {path} 已创建。")
    else:
        print(f"目录 {path} 已存在。")

class AMPLoss(nn.Module):
    def __init__(self):
        super(AMPLoss, self).__init__()
        self.cri = nn.L1Loss()

    def forward(self, x, y):
        x = torch.fft.rfft2(x, norm='backward')
        x_mag = torch.abs(x)
        y = torch.fft.rfft2(y, norm='backward')
        y_mag = torch.abs(y)

        return self.cri(x_mag, y_mag)


class PhaLoss(nn.Module):
    def __init__(self):
        super(PhaLoss, self).__init__()
        self.cri = nn.L1Loss()

    def forward(self, x, y):
        x = torch.fft.rfft2(x, norm='backward')
        x_mag = torch.angle(x)
        y = torch.fft.rfft2(y, norm='backward')
        y_mag = torch.angle(y)

        return self.cri(x_mag, y_mag)        
        
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("训练文件夹为：{}".format(path))
    else:
        print('已存在{}'.format(path))

def findLastCheckpoint(save_dir):
    # 根据匹配模式查找所有的文件名，*、[]、?三种常用通配符
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            # 捕获模型文件后面跟的数字，()中为捕获组
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

def get_filename_list(path, shuffle=False):
    filename_list = os.listdir(path)
    filename_list = [item for item in filename_list if os.path.isfile(os.path.join(path, item))]
    if shuffle:
        random.shuffle(filename_list)
    return filename_list

def fspecial(func_name, kernel_size, sigma):
    if func_name == 'gaussian':
        m = n = (kernel_size - 1.) / 2.
        y, x = ogrid[-m:m + 1, -n:n + 1]
        h = exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

# 高斯滤波模糊图像后，做下采样
def Gaussian_downsample(x, psf, s):
    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)
    y = np.zeros((x.shape[0], int(x.shape[1] / s), int(x.shape[2] / s)))
    for i in range(x.shape[0]):
        x1 = x[i, :, :]
        # mode为same表示输入大小和输出大小一致，boundary表示处理图像边缘的方法，symm表示镜像的重复边缘值
        x2 = signal.convolve2d(x1, psf, boundary='symm', mode='same')
        # 采用跨步连接做下采样
        y[i, :, :] = x2[0::s, 0::s]
    return y


def create_F():
    F = np.array(
        [[2.0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
         [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 11, 16, 19, 21, 20, 18, 16, 14, 11, 7, 5, 3, 2, 2, 1, 1, 2, 2, 2, 2, 2],
         [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16, 9, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    for band in range(3):
        div = np.sum(F[band][:])
        for i in range(31):
            F[band][i] = F[band][i] / div
    return F


def warm_lr_scheduler(optimizer, init_lr1, init_lr2, min, warm_iter, iteraion, lr_decay_iter, max_iter, power):
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer
    if iteraion < warm_iter:
        lr = init_lr1 + iteraion / warm_iter * (init_lr2 - init_lr1)
    else:
        lr = init_lr2 * (1 - (iteraion - warm_iter) / (max_iter - warm_iter)) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lrs

    if lr < min:
        lr = min
    return lr

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


loss_func = nn.L1Loss(reduction='mean').cuda()


def reconstruction(net2, R, HSI_LR, MSI, HRHSI, downsample_factor, training_size, stride, val_loss):
    index_matrix = torch.zeros((R.shape[1], MSI.shape[2], MSI.shape[3])).cuda()
    abundance_t = torch.zeros((R.shape[1], MSI.shape[2], MSI.shape[3])).cuda()
    # 收集每一块的起始横纵坐标
    a = []
    for j in range(0, MSI.shape[2] - training_size + 1, stride):
        a.append(j)
    a.append(MSI.shape[2] - training_size)
    b = []
    for j in range(0, MSI.shape[3] - training_size + 1, stride):
        b.append(j)
    b.append(MSI.shape[3] - training_size)
    for j in a:
        for k in b:
            temp_hrms = MSI[:, :, j:j + training_size, k:k + training_size]
            # ?
            temp_lrhs = HSI_LR[:, :, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                        int(k / downsample_factor):int((k + training_size) / downsample_factor)]
            temp_hrhs = HRHSI[:, :, j:j + training_size, k:k + training_size]
            with torch.no_grad():
                _,_,out = net2(temp_hrms, temp_lrhs)
                assert torch.isnan(out).sum() == 0

                loss_temp = loss_func(out, temp_hrhs.cuda())
                val_loss.update(loss_temp)
                # 去掉维数为一的维度
                HSI = out.squeeze()
                # 将张量中的元素限制在[0,1]范围内
                HSI = torch.clamp(HSI, 0, 1)
                abundance_t[:, j:j + training_size, k:k + training_size] = abundance_t[:, j:j + training_size,
                                                                           k:k + training_size] + HSI
                index_matrix[:, j:j + training_size, k:k + training_size] = 1 + index_matrix[:, j:j + training_size,
                                                                                k:k + training_size]

    HSI_recon = abundance_t / index_matrix
    assert torch.isnan(HSI_recon).sum() == 0
    return HSI_recon, val_loss


