import torch
# import hdf5storage as h5
import scipy.io as sio
from torch.utils.data import Dataset
import numpy as np
from utils import *


class CaveDataset(Dataset):
    def __init__(self, path, R, training_size, stride, downsample_factor, PSF, num):
        """
        :param path:
        :param R: 光谱响应矩阵
        :param training_size:
        :param stride:
        :param downsample_factor:
        :param PSF: 高斯模糊核
        :param num: cave=32个场景
        """
        imglist = os.listdir(path)
        train_hrhs = []
        train_hrms = []
        train_lrhs = []

        for i in range(num):
            data_path = os.path.join(path, imglist[i])
            img = sio.loadmat(data_path)
            img1 = img["b"]
            # img1 = img1 / img1.max()

            HRHSI = np.transpose(img1, (2, 0, 1))
            # hwc-chw
            HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)
            MSI = np.tensordot(R, HRHSI, axes=([1], [0]))
            for j in range(0, HRHSI.shape[1] - training_size + 1, stride):
                for k in range(0, HRHSI.shape[2] - training_size + 1, stride):
                    temp_hrhs = HRHSI[:, j:j + training_size, k:k + training_size]
                    temp_hrms = MSI[:, j:j + training_size, k:k + training_size]
                    # 由于HSI_LR的尺寸是HSI_HR的downsample_factor倍，需要将j和k的索引除以downsample_factor
                    # 这个操作保证了低分辨率图像中提取的patch和高分辨率图像中提取的patch在空间位置上对应
                    temp_lrhs = HSI_LR[:, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                                int(k / downsample_factor):int((k + training_size) / downsample_factor)]
                    # print(temp_hrhs.shape,temp_lrhs.shape,temp_hrms.shape)
                    temp_hrhs=temp_hrhs.astype(np.float32)
                    temp_hrms=temp_hrms.astype(np.float32)
                    temp_lrhs=temp_lrhs.astype(np.float32)
                    train_hrhs.append(temp_hrhs)
                    train_hrms.append(temp_hrms)
                    train_lrhs.append(temp_lrhs)
        train_hrhs = torch.Tensor(np.array(train_hrhs))
        train_hrms = torch.Tensor(np.array(train_hrms))
        train_lrhs = torch.Tensor(np.array(train_lrhs))

        # print(train_hrhs.shape, train_hrms.shape)
        self.train_hrhs_all = train_hrhs
        self.train_hrms_all = train_hrms
        self.train_lrhs_all = train_lrhs

    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_hrms = self.train_hrms_all[index, :, :, :]
        train_lrhs = self.train_lrhs_all[index, :, :, :]
        # print(train_hrhs.shape, train_hrms.shape,train_lrhs.shape)
        return train_hrhs, train_hrms, train_lrhs

    def __len__(self):
        return self.train_hrhs_all.shape[0]