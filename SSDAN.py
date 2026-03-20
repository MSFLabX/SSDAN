import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from torch import einsum
import warnings
from einops import rearrange
# import torch.nn.PixelUnshuffle

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



import math
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, hidden_dim=48, act_layer=nn.GELU, use_eca=False):
        super().__init__()
        hidden_features = dim
        #self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
        #                            act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, hidden_features, kernel_size=3, stride=1, padding=1),
            act_layer())
        #self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        #self.hidden_dim = hidden_dim
        self.eca = eca_layer_1d(hidden_features) if use_eca else nn.Identity()
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(hidden_features, dim, kernel_size=3, stride=1, padding=1),
            act_layer())

    def forward(self, x):
        x = self.dwconv(x)
        x = self.eca(x)
        #x = self.dwconv1(x)
        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(2)
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.pixel_unshuffle(self.body(x))


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, transpose=False,
                 channel_shuffle_g=0, norm_method=nn.BatchNorm2d, groups=1):
        super(BasicConv, self).__init__()
        self.channel_shuffle_g = channel_shuffle_g
        self.norm = norm
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        elif relu:
            layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class SSRB(nn.Module):
    def __init__(self, dim, window_size=(8, 8), dim_head=28, heads=1):
        super().__init__()

        self.pos = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.SARB = ERB(dim, window_size, dim_head, heads)
        self.SRB = ARB(dim)

    def forward(self, x):

        x = self.pos(x) + x
        x = self.SARB(x)
        x = self.SRB(x)

        return x

class ERB(nn.Module):
    def __init__(self, dim, window_size=(8, 8), dim_head=28, heads=1):
        super().__init__()
        self.WSSA = PreNorm(dim, WSSA(dim=dim, window_size=window_size, dim_head=dim_head, heads=heads,
                                      shift=False))
        self.FFN = PreNorm(dim, FFN(dim=dim), norm_type='gn')

    def forward(self, x):

        x = self.WSSA(x) + x
        x = self.FFN(x) + x
        return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm_type='ln'):
        super().__init__()
        self.fn = fn
        self.norm_type = norm_type
        if norm_type == 'ln':
            self.norm = nn.LayerNorm(dim)
        else:
            self.norm = nn.GroupNorm(dim, dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, *args, **kwargs):
        if self.norm_type == 'ln':
            x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class WSSA(nn.Module):
    def __init__(self, dim, window_size=(8, 8), dim_head=28, heads=1, shift=False):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        self.shift = shift

        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1)
        self.apply(self.init_weight)

    def init_weight(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def cal_attention(self, x):
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        h1, h2 = h // self.window_size[0], w // self.window_size[1]
        q, k, v = map(lambda t: rearrange(t, 'b c (h1 h) (h2 w) ->b (h1 h2) c (h w)', h1=h1, h2=h2), (q, k, v))
        q *= self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b (h1 h2) c (h w) -> b c (h1 h) (h2 w)', h1=h1, h=h // h1)
        out = self.to_out(out)
        return out

    def forward(self, x):

        w_size = self.window_size
        if self.shift:
            x = x.roll(shifts=w_size[0]//2, dims=2).roll(shifts=w_size[1]//2, dims=3)
        out = self.cal_attention(x)
        if self.shift:
            out = out.roll(shifts=-1*w_size[1]//2, dims=3).roll(shifts=-1*w_size[0]//2, dims=2)
        return out

class FFN(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):

        out = self.net(x)
        return out

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class ARB(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.CMB = PreNorm(dim, CMB(dim=dim))
        self.SAB = PreNorm(dim, SAB(dim=dim), norm_type='gn')

    def forward(self, x):

        x = self.CMB(x) + x
        x = self.SAB(x) + x
        return x

class CMB(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.to_a = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Conv2d(dim, dim, 11, 1, 5, groups=dim, bias=False),
        )
        self.to_v = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1, 1, 0)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def cal_attention(self, x):
        a, v = self.to_a(x), self.to_v(x)
        out = self.to_out(a*v)
        return out

    def forward(self, x):
        out = self.cal_attention(x)
        return out

class SAB(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        self.conv = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.Estimator = nn.Sequential(
            nn.Conv2d(dim, 1, 3, 1, 1, bias=False),
            GELU(),
        )
        self.SW = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.Sigmoid(),
        )
        self.out = nn.Conv2d(dim, dim, 1, 1, 0)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight.data, mean=0.0, std=.02)

    def forward(self, f):
        f = self.conv(f)
        out = self.SW(f) * self.Estimator(f).repeat(1, self.dim, 1, 1)
        out = self.out(out)
        return out

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class Freprocess(nn.Module):
    def __init__(self, channels):
        super(Freprocess, self).__init__()
        self.pre1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.pre2 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.amp_fuse = nn.Sequential(nn.Conv2d(2 * channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(2 * channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.post = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, msf, panf):
        _, _, H, W = msf.shape
        msF = torch.fft.rfft2(self.pre1(msf) + 1e-8, norm='backward')
        panF = torch.fft.rfft2(self.pre2(panf) + 1e-8, norm='backward')
        # print(panF.shape)
        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)
        panF_amp = torch.abs(panF)
        panF_pha = torch.angle(panF)
        amp_fuse = self.amp_fuse(torch.cat([msF_amp, panF_amp], 1))
        pha_fuse = self.pha_fuse(torch.cat([msF_pha, panF_pha], 1))

        real = amp_fuse * torch.cos(pha_fuse) + 1e-8
        imag = amp_fuse * torch.sin(pha_fuse) + 1e-8
        out = torch.complex(real, imag) + 1e-8
        # print(out.shape)
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))
        # print(out.shape)
        return self.post(out)


class STFreprocess(nn.Module):
    def __init__(self, channels):
        super(STFreprocess, self).__init__()
        self.pre1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.pre2 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.amp_fuse1 = nn.Sequential(nn.Conv2d(2 * channels, 8 * channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                       nn.Conv2d(8 * channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                       nn.Conv2d(channels, channels, 1, 1, 0))
        self.pha_fuse1 = nn.Sequential(nn.Conv2d(2 * channels, 8 * channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                       nn.Conv2d(8 * channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                       nn.Conv2d(channels, channels, 1, 1, 0))
        self.amp_fuse2 = nn.Sequential(nn.Conv2d(2 * channels, 8 * channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                       nn.Conv2d(8 * channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                       nn.Conv2d(channels, channels, 1, 1, 0))
        self.pha_fuse2 = nn.Sequential(nn.Conv2d(2 * channels, 8 * channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                       nn.Conv2d(8 * channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                       nn.Conv2d(channels, channels, 1, 1, 0))
        self.amp_fuse3 = nn.Sequential(nn.Conv2d(2 * channels, 8 * channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                       nn.Conv2d(8 * channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                       nn.Conv2d(channels, channels, 1, 1, 0))
        self.pha_fuse3 = nn.Sequential(nn.Conv2d(2 * channels, 8 * channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                       nn.Conv2d(8 * channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                       nn.Conv2d(channels, channels, 1, 1, 0))
        self.amp_fuse4 = nn.Sequential(nn.Conv2d(2 * channels, 8 * channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                       nn.Conv2d(8 * channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                       nn.Conv2d(channels, channels, 1, 1, 0))
        self.pha_fuse4 = nn.Sequential(nn.Conv2d(2 * channels, 8 * channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                       nn.Conv2d(8 * channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                       nn.Conv2d(channels, channels, 1, 1, 0))


        self.post = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, msf, panf):
        _, _, H, W = msf.shape
        # s1 = 43
        # s2 = 42
        s1 = H // 3
        s2 = W // 3 * 2 + 1


        msfa = msf[:, :, :s2, :s2]
        msfb = msf[:, :, :s2, s1:]
        msfc = msf[:, :, s1:, :s2]
        msfd = msf[:, :, s1:, s1:]
        msf_list = [msfa, msfb, msfc, msfd]



        panfa = panf[:, :, :s2, :s2]
        panfb = panf[:, :, :s2, s1:]
        panfc = panf[:, :, s1:, :s2]
        panfd = panf[:, :, s1:, s1:]
        panf_list = [panfa, panfb, panfc, panfd]


        qH = H // 4
        qW = W // 4

        amp_list = [self.amp_fuse1, self.amp_fuse2, self.amp_fuse3, self.amp_fuse4]
        pha_list = [self.pha_fuse1, self.pha_fuse2, self.pha_fuse3, self.pha_fuse4]

        out_list = [0, 0, 0, 0]
        out_shift = [0, 0, 0, 0]

        for i in range(len(msf_list)):
            h = msf_list[i].shape[-2]
            w = msf_list[i].shape[-1]
            msF = torch.fft.rfft2(self.pre1(msf_list[i]) + 1e-8, norm='backward')
            panF = torch.fft.rfft2(self.pre2(panf_list[i]) + 1e-8, norm='backward')
            # print(msF.shape)
            msF_amp = torch.abs(msF)
            msF_pha = torch.angle(msF)
            panF_amp = torch.abs(panF)
            panF_pha = torch.angle(panF)

            # 不共享权重

            amp_fuse = amp_list[i](torch.cat([msF_amp, panF_amp], 1))
            pha_fuse = pha_list[i](torch.cat([msF_pha, panF_pha], 1))

            real = amp_fuse * torch.cos(pha_fuse) + 1e-8
            imag = amp_fuse * torch.sin(pha_fuse) + 1e-8
            out_list[i] = torch.complex(real, imag) + 1e-8
  
            out_list[i] = torch.abs(torch.fft.irfft2(out_list[i], s=(h, w), norm='backward'))


        outup = torch.cat((out_list[0][:, :, :s1, :s1],
                           (out_list[0][:, :, :s1, s1:] + out_list[1][:, :, :s1, :s2 - s1]) / 2,
                           out_list[1][:, :, :s1, s2 - s1:]
                           ), dim=-1)

        outmid = torch.cat(((out_list[0][:, :, s1:s2, :s1] + out_list[2][:, :, :s2 - s1, :s1]) / 2,
                            (out_list[0][:, :, s1:s2, s1:] + out_list[1][:, :, s1:s2, :s2 - s1] + out_list[2][:, :,:s2 - s1, s1:] + out_list[3][:,:,:s2 - s1,:s2 - s1]) / 4,
                            (out_list[1][:, :, s1:s2, s2-s1:] + out_list[3][:, :, :s2-s1, s2-s1:]) / 2), dim=-1)

        outdown = torch.cat((out_list[2][:, :, s2 - s1:, :s1],
                             (out_list[2][:, :, s2 - s1:, s1:] + out_list[3][:, :, s2 - s1:, :s2 - s1]) / 2,
                             out_list[3][:, :, s2-s1:, s2 - s1:]
                             ), dim=-1)

        out = torch.cat((outup, outmid, outdown), dim=-2)

        return self.post(out)

def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

def mean_channels(F):
    assert (F.dim() == 4)
    # 依次沿着第四个维度和第三个维度求和(不改变图像的维度大小)，获得每个通道的和
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    # 计算每个通道上的均值
    return spatial_sum / (F.size(2) * F.size(3))


class SpaFreInterFusion(nn.Module):
    def __init__(self, channels):
        super(SpaFreInterFusion, self).__init__()

        self.spa_preprocess = nn.Conv2d(channels, channels, 3, 1, 1)
        self.fre_preprocess = nn.Conv2d(channels, channels, 3, 1, 1)

        self.weight_spa = nn.Sequential(nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True),
                                    nn.ReLU(),
                                    nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True),
                                    nn.Sigmoid())

        self.weight_fre = nn.Sequential(nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True),
                                        nn.ReLU(),
                                        nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True),
                                        nn.Sigmoid())

        self.spa_spa_diff_fusion = nn.Sequential(nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1, bias=True),
                                    nn.ReLU(),
                                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True))

        self.fre_spa_diff_fusion = nn.Sequential(nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1, bias=True),
                                                 nn.ReLU(),
                                                 nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True))


    def channel_cosine_similarity(self, img1, img2):
        # 前文的通道维度余弦相似度计算函数
        B, C, H, W = img1.shape
        N = H * W
        img1_flat = img1.reshape(B, C, N).float()
        img2_flat = img2.reshape(B, C, N).float()
        img1_norm = torch.nn.functional.normalize(img1_flat, p=2, dim=-1)
        img2_norm = torch.nn.functional.normalize(img2_flat, p=2, dim=-1)
        similarity = torch.sum(img1_norm * img2_norm, dim=-1)
        return similarity  # 输出[B, C]

    def forward(self, spa, fre, spa_raw, spe_raw):
        _,_,H,W = spa.shape
        spa_1 = self.spa_preprocess(spa)
        fre_1 = self.fre_preprocess(fre)

        pixel_similarity_spa = spa_1 * spa_raw
        pixel_similarity_spa = pixel_similarity_spa.sum(dim=1)

        pixel_similarity_spa = pixel_similarity_spa.unsqueeze(1)
        pixel_similarity_spa_weight = self.weight_spa(pixel_similarity_spa)
        pixel_similarity_spa_weight = 1 - pixel_similarity_spa_weight
        spa_diff_spa = spa_raw * pixel_similarity_spa_weight


        pixel_similarity_fre = fre_1 * spa_raw
        pixel_similarity_fre = pixel_similarity_fre.sum(dim=1)

        pixel_similarity_fre = pixel_similarity_fre.unsqueeze(1)
        pixel_similarity_fre_weight = self.weight_fre(pixel_similarity_fre)
        pixel_similarity_fre_weight = 1 - pixel_similarity_fre_weight
        fre_diff_spa = spa_raw * pixel_similarity_fre_weight

        spa_spe_channel_weight = self.channel_cosine_similarity(spa_1, spe_raw)
        spa_spe_channel_weight = spa_spe_channel_weight.unsqueeze(-1).unsqueeze(-1)
        spa_spe_channel_diff_weight = 1 - spa_spe_channel_weight
        spa_spe_diff = spe_raw * spa_spe_channel_diff_weight

        fre_spe_channel_weight = self.channel_cosine_similarity(fre_1, spe_raw)
        fre_spe_channel_weight = fre_spe_channel_weight.unsqueeze(-1).unsqueeze(-1)
        fre_spe_channel_diff_weight = 1 - fre_spe_channel_weight
        fre_spe_diff = spe_raw * fre_spe_channel_diff_weight

    
        spa_fusion_1 = spa + spa_spe_diff
        spa_fusion_2 = self.spa_spa_diff_fusion(torch.cat([spa_fusion_1, spa_diff_spa], dim=1))

        fre_fusion_1 = fre + fre_spe_diff
        fre_fusion_2 = self.fre_spa_diff_fusion(torch.cat([fre_fusion_1, fre_diff_spa], dim=1))
        return spa_fusion_2, fre_fusion_2

def fft(X):
    # 对输入X进行FFT变换
    F_X = torch.fft.fftn(X, dim=(-2, -1))
    # 全局滤波
    return F_X


def high_pass_filter(F_X, percent=0.1):
    # 计算低频部分的数量
    H, W = F_X.shape[-2:]
    num_low_freq = int(H * W * percent)

    # 计算频率坐标
    u = torch.fft.fftfreq(H)[:, None]
    v = torch.fft.fftfreq(W)[None, :]
    freq_dist = torch.sqrt(u ** 2 + v ** 2)

    # 找到低频部分的索引
    sorted_indices = torch.argsort(freq_dist.flatten())
    low_freq_indices = sorted_indices[:num_low_freq]

    # 创建HPF掩码
    mask = torch.ones_like(F_X)
    # low_freq_indices // W计算低频部分在二维mask中的行坐标，low_freq_indices % W计算低频部分在二维mask中的列坐标
    # 将低频部分的掩码置为0
    mask[..., low_freq_indices // W, low_freq_indices % W] = 0

    # 应用HPF
    h_F_X = F_X * mask
    return h_F_X


def inverse_fft(F_X):
    # 进行逆FFT变换
    X = torch.fft.ifftn(F_X, dim=(-2, -1)).real
    return X


class FrequencyAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(FrequencyAttentionModule, self).__init__()
        # 可学习的全局滤波器核
        # 局部分支的1x1卷积层
        self.local_branch = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, X):
        # 全局分支
        g_X_K = fft(X)
        h_g_X_K = high_pass_filter(g_X_K)
        F_global = inverse_fft(h_g_X_K)

        return F_global

class SpaFremid(nn.Module):
    def __init__(self, channels, window_size=(8, 8)):
        super(SpaFremid, self).__init__()
        self.panprocess = nn.Conv2d(channels, channels, 3, 1, 1)
        self.speprocess = nn.Conv2d(channels, channels, 3, 1, 1)
        # self.panpre = nn.Conv2d(channels, channels, 1, 1, 0)
        self.fre_process = Freprocess(channels)
        self.STfre_process = STFreprocess(channels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.contrast = stdv_channels
        self.cha_att1 = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=1, padding=0, bias=True),
                                      nn.LeakyReLU(0.1),
                                      nn.Conv2d(channels // 2, channels, kernel_size=1, padding=0, bias=True),
                                      nn.Sigmoid())
        self.cha_att2 = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=1, padding=0, bias=True),
                                      nn.LeakyReLU(0.1),
                                      nn.Conv2d(channels // 2, channels, kernel_size=1, padding=0, bias=True),
                                      nn.Sigmoid())
        self.FT_ST_fusion = nn.Sequential(nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, bias=True),
                                           nn.LeakyReLU(0.1),
                                           nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=True))
        self.FT_CF_pre_fusion = nn.Sequential(nn.Conv2d(2 * channels, 2 * channels, kernel_size=3, padding=1, bias=True),
                                              nn.LeakyReLU(0.1),
                                              nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1, bias=True))
        self.SPa_CF_pre_fusion = nn.Sequential(nn.Conv2d(2 * channels, 2 * channels, kernel_size=3, padding=1, bias=True),
                                              nn.LeakyReLU(0.1),
                                              nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1, bias=True))
        self.post1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.post2 = nn.Conv2d(channels, channels, 3, 1, 1)

        self.win_attention = SSRB(channels, window_size=window_size)

        self.spa_preprocess = nn.Sequential(nn.Conv2d(2 * channels, 2 * channels, kernel_size=3, padding=1, bias=True),
                                            nn.LeakyReLU(0.1),
                                            nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1, bias=True))


        # self.channel_fft = Channel_FFT_Process(64)
        self.texture_enchance = FrequencyAttentionModule(channels)

        self.SpaFreInterFusion = SpaFreInterFusion(channels)

    # msst:全局频率融合后的输出  msft:局部频率融合后的输出  pan:全色特征图
    def forward(self, input):  # , i  实际输入是 msst msft
        msst = input[1]
        msft = input[0]

        out_msi = self.panprocess(input[2])

        out_hsi = self.speprocess(input[3])

        frefuse = self.fre_process(msst, msft)
        STfuse = self.STfre_process(msst, msft)

        panf_h = self.texture_enchance(out_msi)

        # muti-fusion
        spa_preposs = self.spa_preprocess(torch.cat([msst, msft], 1))
        spafuse = self.win_attention(spa_preposs)

        FT_fused = self.FT_ST_fusion(torch.cat([STfuse, frefuse], dim=1))


        spa_out, fre_out = self.SpaFreInterFusion(spafuse, FT_fused, out_msi, out_hsi)


        # 空间和通道傅里叶融合
        ms_SPa = self.SPa_CF_pre_fusion(torch.cat([spa_out, panf_h], dim=1))
        # 频域和通道傅里叶融合
        ms_FT = self.FT_CF_pre_fusion(torch.cat([fre_out, panf_h], dim=1))

        fin_SPa = self.post1(self.cha_att1(self.contrast(ms_SPa) + self.avgpool(ms_SPa)) * ms_SPa)
        fin_FT = self.post2(self.cha_att2(self.contrast(ms_FT) + self.avgpool(ms_FT)) * ms_FT)

        out_SPa = fin_SPa + msst
        out_FT = fin_FT + msft

        return {1: out_SPa, 0: out_FT, 2: out_msi, 3: out_hsi}

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        inp_channels = 130
        dim = 48
        num_blocks = [1, 1, 1, 1]
        heads = [1, 1, 1, 1]
        ffn_expansion_factor = 2.66
        bias = False
        LayerNorm_type = 'WithBias'
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.upSample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.Conv96_31 = nn.Conv2d(in_channels=96, out_channels=31, kernel_size=3, padding=(1, 1))
        self.Conv192_31 = nn.Conv2d(in_channels=192, out_channels=31, kernel_size=3, padding=(1, 1))
        self.re = nn.ReLU(inplace=True).cuda()
        self.Conv144_96 = nn.Conv2d(in_channels=144, out_channels=96, kernel_size=3, padding=(1, 1))
        self.Conv240_192 = nn.Conv2d(in_channels=240, out_channels=192, kernel_size=3, padding=(1, 1))
        self.Conv1 = nn.Conv2d(in_channels=240, out_channels=192, kernel_size=3, padding=(1, 1))
        self.encoder_level11 = nn.Sequential(SpaFremid(channels=dim),
                                             SpaFremid(channels=dim),
                                             SpaFremid(channels=dim),
                                             SpaFremid(channels=dim))
        self.Conv3_64 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=3, padding=(1, 1))
        self.Conv31_64 = nn.Conv2d(in_channels=31, out_channels=48, kernel_size=3, padding=(1, 1))



    # x:MSI, y:HSI
    def forward(self, x, y):
        # 高光谱图像上采样为原来的8倍
        y = self.upSample(y)
        # 多光谱图像通道数由3卷积成48
        X = self.Conv3_64(x)
        # 高光谱图像通道数由31卷积成48
        Y = self.Conv31_64(y)
        #X = self.encoder_level1(X)
        #Y = self.encoder_level1(Y)
        # hr_msi1 = X.clone()
        # 构造输入字典
        XX1 = {0: X, 1: Y, 2: X, 3: Y}
        # 输入Transformer编码器
        output1 = self.encoder_level11(XX1)
        X, Y = output1[0], output1[1]
        # 将输入和输出叠加起来，通道数为48+48+3+31
        Z = torch.cat([X, Y, x, y], dim=1)

        #下采样2倍
        x1 = F.interpolate(x, scale_factor=0.5)
        y1 = F.interpolate(y, scale_factor=0.5)
        # 将通道数由3卷积成48
        X1 = self.Conv3_64(x1)
        # 将通道数由31卷积成48
        Y1 = self.Conv31_64(y1)
        # hr_msi2 = X1.clone()

        XX2 = {0: X1, 1: Y1, 2: X1, 3: Y1}
        # 输入Transformer编码器
        output2 = self.encoder_level11(XX2)
        X1, Y1 = output2[0], output2[1]
        # 将输入输出叠加起来，通道数为48+48+3+31
        Z1 = torch.cat([X1, Y1, x1, y1], dim=1)
        inp_enc_level11 = self.patch_embed(Z1)

        #下采样4倍
        x2 = F.interpolate(x1, scale_factor=0.5)
        y2 = F.interpolate(y1, scale_factor=0.5)
        X2 = self.Conv3_64(x2)
        Y2 = self.Conv31_64(y2)

        # hr_msi3 = X2.clone()
        #X2 = self.encoder_level1(X2)
        #Y2 = self.encoder_level1(Y2)
        XX3 = {0: X2, 1: Y2, 2: X2, 3: Y2}
        output3 = self.encoder_level11(XX3)
        X2, Y2 = output3[0], output3[1]
        Z2 = torch.cat([X2, Y2, x2, y2], dim=1)
        inp_enc_level12 = self.patch_embed(Z2)

        inp_enc_level1 = self.patch_embed(Z)

        # 做2倍下采样，通道数变成原来的2倍，
        inp_enc_level2 = self.down1_2(inp_enc_level1)
        # 下采样2倍后使得空间大小一致后沿通道叠加，通道数为96+48
        ZZ1 = torch.cat([inp_enc_level2, inp_enc_level11], dim=1)
        # 通道数调整为96
        ZZ1 = self.Conv144_96(ZZ1)

        # 下采样2倍后使得空间大小一致后沿通道叠加
        inp_enc_level3 = self.down2_3(ZZ1)
        # 通道数为192+48
        ZZ2 = torch.cat([inp_enc_level3, inp_enc_level12], dim=1)
        # 通道数调整为192
        ZZ2 = self.Conv240_192(ZZ2)
        # 通道数调整为31
        X1 = self.Conv192_31(ZZ2)
        # 和四倍下采样的值做残差
        X1 = y2+X1

        out_dec_level3 = ZZ2
        # 上采样2倍，通道数调整为96
        inp_dec_level2 = self.up3_2(out_dec_level3)
        # 和对应尺度的值叠加，通道数变成192
        inp_dec_level2 = torch.cat([inp_dec_level2, ZZ1], 1)
        # 调整通道数为96
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        # 调整通道数为31
        X2 = self.Conv96_31(inp_dec_level2)
        # 和2倍下采样的值做残差
        X2 = y1+X2
        # 上采样2倍，通道数变成48
        inp_dec_level1 = self.up2_1(inp_dec_level2)
        # 通道维度叠加起来，通道数变成96
        inp_dec_level1 = torch.cat([inp_dec_level1, inp_enc_level1], 1)
        # 通道数由96卷积成31
        X = self.Conv96_31(inp_dec_level1)
        # 和原始的输入做残差
        X = X + y
        # 通过简单的激活函数做重构
        X = self.re(X)
        return X1, X2, X
