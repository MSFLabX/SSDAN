import logging
import yaml
import sys
sys.path.append('/xiaolin/yt/hsfusion/HFIN')
from calculate_metrics import Loss_SAM, Loss_RMSE, Loss_PSNR, Loss_SSIM, Loss_ERGAS
from thop import profile, clever_format
from CaveDataset import CaveDataset
from utils import *
from SSDAN import Net
from utils import make_dir
from scipy.io import loadmat
from torch import nn
from tqdm import tqdm
import time
import pandas as pd
import torch.utils.data as data
import math
import torch.nn.functional as F
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


if __name__ == '__main__':
    # test_path = '../../data/Cave/Test/HSI'
    # test_filename = 'jelly_beans.mat'
    # test_file_path = os.path.join(test_path, test_filename)
    # img = loadmat(test_file_path)
    # img1 = img["hsi"]
    # print(img1.shape)
    # exit(0)
    fe = open('cfg.yaml')
    cfg = yaml.safe_load(fe)
    save_train_path = '../Checkpoint/model/SSDAN/train'
    save_test_path = '../Checkpoint/model/SSDAN/test'
    record_path = '../record/model/SSDAN'
    train_path = '../GS/DSPNet/Data/CAVE/Train/HSI'
    test_path = '../GS/DSPNet/Data/CAVE/Test/HSI'
    train_filename_list = get_filename_list(train_path)
    test_filename_list = get_filename_list(test_path)

    # 训练参数
    # 损失函数移动到GPU上
    l1_loss = nn.L1Loss(reduction='mean').cuda()
    # loss_func = nn.MSELoss(reduction='mean')

    R = create_F()
    PSF = fspecial('gaussian', 8, 3)
    downsample_factor = cfg['CAVE']['train']['downsample_factor']
    training_size = cfg['CAVE']['train']['image_size']
    train_stride = cfg['CAVE']['train']['stride']
    test_stride = cfg['CAVE']['test']['stride']
    lr = cfg['CAVE']['train']['lr']
    # print(type(lr))
    end_epoch = cfg['CAVE']['train']['epoch']
    end_epoch = 900
    print(end_epoch)
    weight_decay = cfg['CAVE']['train']['weight_decay']
    batch_size = cfg['CAVE']['train']['batch_size']
    num = cfg['CAVE']['train']['image_num']
    psnr_optimal = 40
    rmse_optimal = 2

    test_epoch = 100
    val_interval = 50           # 每隔val_interval epoch测试一次
    checkpoint_interval = 100
    # 计算总的迭代次数，(512 - training_size) // stride + 1) ** 2为一张图片的迭代次数，*num为20张图片的迭代次数
    # /BATCH_SIZE为一个epoch的迭代次数
    maxiteration = math.ceil(((512 - training_size) // train_stride + 1) ** 2 * num / batch_size) * end_epoch
#     maxiteration = math.ceil(
#          ((1040 - training_size) // train_stride + 1) * ((1392 - training_size) // train_stride + 1) * num / batch_size) * end_epoch
    print("maxiteration：", maxiteration)


    # warm_lr_scheduler
    decay_power = 1.5
    init_lr2 = 2e-4
    init_lr1 = 2e-4 / 10
    min_lr = 0
    warm_iter = math.floor(maxiteration / 40)

    make_dir(record_path)
    df = pd.DataFrame(columns=['epoch', 'lr', 'train_loss', 'val_loss', 'val_rmse', 'val_psnr', 'val_sam', 'val_ssim', 'val_ergas'])  # 列名
    data_name = 'cave'
    excel_name = data_name+'_record.csv'
    excel_path = os.path.join(record_path, excel_name)
    if not os.path.exists(excel_path):
        df.to_csv(excel_path, index=False)

    train_dataset = CaveDataset(train_path, R, training_size, train_stride, downsample_factor, PSF, num)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    a,b=64,64
    hsi = torch.randn(2, 31,a//8,b//8).cuda()
    msi = torch.randn(2, 3, a,b).cuda()
    model2 = Net().cuda()
    flops, params,DIC= profile(model2, inputs=(msi,hsi),ret_layer_info=True)
    flops, params= clever_format([flops, params], "%.3f")
    print(flops, params)
    

    model = Net().cuda()
    # 模型初始化
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            # 初始化LayerNorm层
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # optimizer = torch.optim.Adam([{'params': cnn.parameters(), 'initial_lr': 1e-1}], lr=LR,betas=(0.2, 0.999),weight_decay=weight_decay)
    # weight_decay表示在每次参数更新时，参数将乘以(1 - learning_rate * weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    # 创建学习率调度器，按照余弦函数得形式调整学习率，通常用于在训练过程中平滑得减少学习率，从而帮助模型在训练后期更细致得逼近最优解
    # T_max=maxiteration为训练过程中的总迭代次数，学习率调度器将根据这个值来计算每个迭代的学习率
    # eta_min为训练结束时的最小学习率，默认值通常为0，这里设置为1e-6
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, maxiteration, eta_min=1e-6)
    # start_epoch = 0
    start_epoch = findLastCheckpoint(save_dir=save_train_path)

    if start_epoch > 0:
        print('resuming by loading epoch %04d' % start_epoch)
        config_parameter = torch.load(os.path.join("../Checkpoint/model/SSDAN/train", 'model_%04d.pth' % start_epoch))
        model.load_state_dict(config_parameter['net_parameter'])
        optimizer.load_state_dict(config_parameter['optimizer_parameter'])
        scheduler.load_state_dict(config_parameter['scheduler_parameter'])

    # resume = True
    resume = False
    # torch.autograd.set_detect_anomaly(True)
    # step = start_step
    step = 0   # warm_lr_scheduler要用
    for epoch in range(start_epoch+1, end_epoch+1):
        model.train()
        loss_all = []
        epoch_loss = 0
        loop = tqdm(train_loader, total=len(train_loader))
        start_time = time.time()
        #for epoch_step, (a1, a2,a3) in enumerate(loop):
        for hr_hsi, hr_msi, lr_hsi in loop:
            lr = optimizer.param_groups[0]['lr']
            step = step + 1
            hr_hsi = hr_hsi.cuda()
            hr_msi = hr_msi.cuda()
            lr_hsi = lr_hsi.cuda()
            output3,output2,output = model(hr_msi, lr_hsi)
            GT2 = F.interpolate(hr_hsi, scale_factor=0.5)
            GT3 = F.interpolate(GT2, scale_factor=0.5)
            
            loss = l1_loss(output, hr_hsi) + l1_loss(output2, GT2) + l1_loss(output3, GT3)
            epoch_loss = epoch_loss + loss.item()
            loss_all.append(np.array(loss.detach().cpu().numpy()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            #loop.set_description(f'Epoch [{epoch}/{EPOCH}]')
            loop.set_description('epoch:{}  lr:{}  loss:{}'.format(epoch, lr, np.mean(loss_all)))
            #loop.set_postfix({'loss': '{0:1.8f}'.format(np.mean(loss_all)), "lr": '{0:1.8f}'.format(lr)})

        elapsed_time = time.time() - start_time
        # print('epcoh = %4d , loss = %.10f , time = %4.2f s' % (epoch + 1, epoch_loss / len(train_dataset), elapsed_time))
        if epoch % cfg['CAVE']['train']['epoch_gap'] == 0:
            checkpoint = {'net_parameter': model.state_dict(), 'optimizer_parameter': optimizer.state_dict(),
                          'scheduler_parameter': scheduler.state_dict(), 'epoch': epoch,
                          'loss': epoch_loss / len(train_dataset)}
            make_dir(save_train_path)
            # logging.info(f'Save checkpoint to models/{cfg["exp_name"]}.pth')
            save_path = os.path.join(save_train_path, 'model_%04d.pth' % epoch)
            # if not os.path.exists(save_train_path):
            #     os.makedirs(save_train_path)
            torch.save(checkpoint, save_path)

        # scheduler.step()

        if ((epoch % val_interval == 0) and (epoch >= test_epoch)) or epoch == 1:
            model.eval()
            val_loss = AverageMeter()
            SAM = Loss_SAM()
            RMSE = Loss_RMSE().cuda()
            PSNR = Loss_PSNR().cuda()
            SSIM = Loss_SSIM().cuda()
            ERGAS = Loss_ERGAS().cuda()
            sam = AverageMeter()
            rmse = AverageMeter()
            psnr = AverageMeter()
            ssim = AverageMeter()
            ergas = AverageMeter()
            with torch.no_grad():
                for i in range(0, len(test_filename_list)):
                    test_file_path = os.path.join(test_path, test_filename_list[i])
                    img = loadmat(test_file_path)
                    img1 = img["b"]
                    img1 = img1 / img1.max()
                    HRHSI = torch.Tensor(np.transpose(img1, (2, 0, 1)))
                    MSI = torch.tensordot(torch.Tensor(R), HRHSI, dims=([1], [0]))
                    HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)
                    MSI_1 = torch.unsqueeze(MSI, 0)
                    HSI_LR1 = torch.unsqueeze(torch.Tensor(HSI_LR), 0)  # 加维度 (b,c,h,w)
                    # 计算val_loss用的，防止出错单独拿出来
                    to_fet_loss_hr_hsi = torch.unsqueeze(torch.Tensor(HRHSI), 0)

                    prediction, val_loss = reconstruction(model, R, HSI_LR1.cuda(), MSI_1.cuda(), to_fet_loss_hr_hsi, downsample_factor, training_size, test_stride, val_loss)
                    # print(Fuse.shape)
                    sam.update(SAM(np.transpose(HRHSI.cpu().detach().numpy(), (1, 2, 0)), np.transpose(prediction.squeeze().cpu().detach().numpy(), (1, 2, 0))))
#                     print(HRHSI.type(), prediction.type())
                    rmse.update(RMSE(HRHSI.cuda().permute(1, 2, 0), prediction.squeeze().permute(1, 2, 0)))
#                     print(HRHSI.type(), prediction.type())
                    psnr.update(PSNR(HRHSI.cuda().permute(1, 2, 0), prediction.squeeze().permute(1, 2, 0)))
                    ssim.update(SSIM(HRHSI.cuda().unsqueeze(0), prediction.unsqueeze(0)))
                    ergas.update(ERGAS(HRHSI.cuda().unsqueeze(0), prediction.unsqueeze(0)))
#                     print(HRHSI.shape)
#                     exit(0)
                make_dir(save_test_path)
                if epoch == 290:
                    torch.save(model.state_dict(), save_test_path + '/' + str(epoch) + 'EPOCH' + '_PSNR_best.pkl')

                if torch.abs(psnr_optimal-psnr.avg) < 0.15:
                    torch.save(model.state_dict(), save_test_path + '/' + str(epoch) + 'EPOCH' + '_PSNR_best.pkl')
                if psnr.avg > psnr_optimal:
                    psnr_optimal = psnr.avg

                if torch.abs(rmse.avg-rmse_optimal) < 0.15:
                    torch.save(model.state_dict(), save_test_path + '/' + str(epoch) + 'EPOCH' + '_RMSE_best.pkl')
                if rmse.avg < rmse_optimal:
                    rmse_optimal = rmse.avg

                # detach()方法将张量从当前的计算图中分离出来，确保张量不需要梯度，numpy()方法将tensor转化为numpy
                print("val  PSNR:", psnr.avg.cpu().detach().numpy(), "  RMSE:", rmse.avg.cpu().detach().numpy(), "  SAM:", sam.avg, "val loss:", val_loss.avg.cpu().detach().numpy(), "ssim:", ssim.avg.cpu().detach().numpy(), "ergas:", ergas.avg)
#                 print(ssim.avg.type(), ergas.avg.type())
                val_list = [epoch, lr, np.mean(loss_all), val_loss.avg.cpu().detach().numpy(), rmse.avg.cpu().detach().numpy(), psnr.avg.cpu().detach().numpy(), sam.avg, ssim.avg.cpu().detach().numpy(), ergas.avg]
                val_data = pd.DataFrame([val_list])
                val_data.to_csv(excel_path, mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了
                time.sleep(0.1)


