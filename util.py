#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import math
import os
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import PIL
import numpy as np
import torch
import torchvision.utils as tvu
from PIL import Image
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from config import BaseConfig
from .datasets import ImgDataset


class Bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# 计算PSNR指标
def _psnr(img1, img2, data_range=1.0):
    """https://zhuanlan.zhihu.com/p/309892873"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100
    if data_range == 255.0:
        return 20 * math.log10(255.0 / math.sqrt(mse))
    elif data_range == 1.0:
        return 20 * math.log10(1.0 / math.sqrt(mse))
    else:
        raise ValueError('data_range must be 1 or 255')


def calculate_rgb_psnr(pred, target, data_range=1.0):
    """rgb三通道图像psnr"""
    assert pred.shape == target.shape  # [b, c, h, w] or [c, h, w]
    if target.dim() == 3:
        n_channels = target.shape[0]
    elif target.dim() == 4:
        n_channels = target.shape[1]
    else:
        raise ValueError('Tensor shape error')

    sum_psnr = 0
    for i in range(n_channels):
        this_psnr = _psnr(pred[..., i, ...], target[..., i, ...], data_range)
        sum_psnr += this_psnr
    return sum_psnr / n_channels


def calculate_metrics(path):
    psnr = 0
    transform = ToTensor()
    if len(os.listdir(path)) % 2 == 0:
        count = len(os.listdir(path)) // 2
    else:
        raise ValueError('images are not a pair')
    for i in range(count):
        target = transform(Image.open(os.path.join(path, f"{i}.png")))
        pred = transform(Image.open(os.path.join(path, f"{i}_pred.png")))
        psnr += calculate_rgb_psnr(pred, target)
    return psnr / count


def save_result(x, file_path):
    """保存预测图像为文件"""
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    tvu.save_image(x, file_path, padding=0)


def write_log(logdir, logdata):
    with open(Path(logdir, "data.json"), mode="w") as f:
        try:
            f.seek(0)
            json.dump(logdata, f)
        except Exception:
            print(Bcolors.FAIL + '\n')
            print(logdata)


def plot_log(
        path: Optional[Union[str, Path]] = None, title: str = "graph",
        x: Optional[List[float]] = None, y: Optional[List[float]] = None,
        xlabel: Optional[str] = None, ylabel: Optional[str] = None,
        x_ticks: Optional[Tuple[float, float, float]] = None, y_ticks: Optional[Tuple[float, float, float]] = None,
        x_range: Optional[Tuple[float, float]] = None, y_range: Optional[Tuple[float, float]] = None,
        linestyles: Optional[List[str]] = None, colors: Optional[List[str]] = None, markers: Optional[List[str]] = None,
        dpi=600, file_format='jpg',
        **kwargs
):
    """绘制折线图到文件。
    Args:
        path: 保存的图像路径 为空则直接显示，为目录路径则以 :attr:`title` 为图像名
        title:  图像的标题（也为文件名）为空则默认 graph.png
        x: x轴坐标值，为空则默认从0开始: (start, end)，不从 0 开始则要传参指定
        y: y轴坐标值，给定了则只画一条线: (start, end)，不从 0 开始则要传参指定
        xlabel: 横坐标标题
        ylabel: 纵坐标标题
        x_ticks: 坐标刻度间隔: (start, end, step)
        y_ticks: 坐标刻度间隔: (start, end, step)
        x_range: 坐标轴显示数据范围: (start, end)，基于 x 给定的数据范围，一般不用
        y_range: 坐标轴显示数据范围: (start, end)，基于 y 给定的数据范围，一般不用
        linestyles: 每条数据线的样式
        colors: 每条数据线的颜色
        markers: 每条数据线节点的样式
        dpi: 保存图像的dpi
        file_format: 图像的格式
        **kwargs: 画图的数据
    """
    markers = markers if markers else ['.', '*', 'v', '+', 'x', ',', 'o']
    linestyles = linestyles if linestyles else ['-', '--', '-.', ':']  # 只有这几种
    colors = colors if colors else ['#F27970', '#BB9727', '#14517C', '#32B897', '#05B9E2', '#8983BF', '#D8383A']
    plt.grid(True)
    if y is not None:
        plt.plot(x, y, linewidth=1.5, linestyle=linestyles[0], color=colors[0], marker=markers[0], markersize=4)
    else:
        for index, (key, value) in enumerate(kwargs.items()):
            index = int(index)
            plt.plot(
                range(len(value)) if x is None else x,
                value,
                label=rf"${key}$",  # 启用 Latex 渲染，文本中的空格会被忽略，解决方法："a bc" -> "a\\bc"，即空格用 "\\" 代替
                linewidth=1.5,
                linestyle=linestyles[index % len(linestyles)],
                color=colors[index % len(colors)],
                marker=markers[index % len(markers)],
                markersize=4
            )
    plt.title(title) if title is not None else ...
    plt.xlabel(f"${xlabel}$") if xlabel is not None else ...
    plt.ylabel(f"${ylabel}$") if ylabel is not None else ...
    plt.xticks(range(*x_ticks)) if x_ticks is not None else ...
    plt.yticks(range(*y_ticks)) if y_ticks is not None else ...
    plt.xlim(x_range) if x_range is not None else ...
    plt.ylim(y_range) if y_range is not None else ...
    plt.legend(fontsize=8)
    plt.tight_layout()
    if path is None:
        plt.show()
    elif Path(path).suffix == '':  # 目录路径
        plt.savefig(Path(path, title + f'.{file_format}'), dpi=dpi)
    else:  # 文件路径
        plt.savefig(Path(path), dpi=dpi)

    plt.close("all")


def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


def get_optimizer(optimizer, parameters, lr=0.0001):
    if optimizer == 'Adam':
        return optim.Adam([{'params': parameters, 'initial_lr': lr}], lr=lr, weight_decay=0.000, betas=(0.9, 0.999), amsgrad=False,
                          eps=0.00000001)
    elif optimizer == 'AdamW':
        return optim.AdamW([{'params': parameters, 'initial_lr': lr}], lr=lr, weight_decay=0.001, betas=(0.9, 0.999), amsgrad=False,
                           eps=0.00000001)
    elif optimizer == 'RMSProp':
        return optim.RMSprop([{'params': parameters, 'initial_lr': lr}], lr=lr, weight_decay=0.000)
    elif optimizer == 'SGD':
        return optim.SGD([{'params': parameters, 'initial_lr': lr}], lr=lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} not understood.'.format(optimizer))


def check_dir_exist(*directorys):
    for directory in directorys:
        Path(directory).mkdir(parents=True, exist_ok=True)


def sigmoid(x):
    return 1 / (np.exp(-x) + 1)


def get_beta_schedule(beta_schedule, beta_start, beta_end, num_timesteps):
    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_timesteps, 1, num_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_timesteps,)
    return betas


def generate_cosine_schedule(T, s=0.008):
    """生成余弦方差表"""

    def f(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2

    alphas = []
    f0 = f(0, T)

    for t in range(T + 1):
        alphas.append(f(t, T) / f0)

    betas = []

    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))

    return np.array(betas)


def generate_linear_schedule(T, low, high):
    """生成线性方差表"""
    return np.linspace(low, high, T)


def get_params(img, output_size, n):
    """划分图片为patch，返回打乱后的patch坐标（左上角）：[i,j,th,tw]

    Args:
        img: 图片
        output_size: 每个patch大小
        n: 划分为多少个patch（意味着划分存在重叠patch）
    """
    b, c, h, w = img.shape
    th, tw = output_size
    if w == tw and h == th:
        # 图片大小即为设定的patch大小，则不需要分割，索引i=j=0
        return 0, 0, h, w

    i_list = [random.randint(0, h - th) for _ in range(n)]
    j_list = [random.randint(0, w - tw) for _ in range(n)]
    return i_list, j_list, th, tw


def n_random_crops(img, x, y, h, w):
    """随机裁减（打乱）图片patch

    传入的x,y已经是被打乱的patch坐标点，直接循环采样即可
    """
    crops = []
    for i in range(len(x)):
        new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
        crops.append(new_crop)
    return tuple(crops)


def concat_img_channel(input_img: torch.Tensor, gt_img: torch.Tensor, parse_patches=False, patch_size=None, n=None):
    if parse_patches:
        assert patch_size is not None and n is not None
        # get_params(img,(64,64),16) => len(i=j=n)=16, h=w=64 将图片划分为patch
        i, j, h, w = get_params(input_img, (patch_size, patch_size), n)
        # n_random_crops(img,len:16,len:16,64,64) 将划分的patch打乱
        input_img = n_random_crops(input_img, i, j, h, w)
        gt_img = n_random_crops(gt_img, i, j, h, w)
        # 将打乱的input patch与对应的gt patch进行通道拼接
        # outputs: 16个形状为[6,64,64]的张量组成的list
        outputs = [torch.cat([input_img[i], gt_img[i]], dim=0) for i in range(n)]
        # stack() 拼接张量将list转为tensor，=> [16,6,64,64]
        return torch.stack(outputs, dim=0)
    else:
        # Resizing images to multiples of 16 for whole-image restoration
        wd_new, ht_new = input_img.size
        if ht_new > wd_new and ht_new > 1024:
            wd_new = int(np.ceil(wd_new * 1024 / ht_new))  # 等比例缩放
            ht_new = 1024
        elif ht_new <= wd_new and wd_new > 1024:
            ht_new = int(np.ceil(ht_new * 1024 / wd_new))
            wd_new = 1024
        wd_new = int(16 * np.ceil(wd_new / 16.0))
        ht_new = int(16 * np.ceil(ht_new / 16.0))
        input_img = input_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
        gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)

        return torch.cat([input_img, gt_img], dim=0)


def print_model_params_(model):
    """输出模型网络层及相应的值和梯度，异常值会突出显示"""
    v_n = []  # name
    v_v = []  # value
    v_g = []  # grad
    for name, parameter in model.named_parameters():
        v_n.append(name)
        v_v.append(parameter.detach().cpu().numpy() if parameter is not None else [0])
        v_g.append(parameter.grad.detach().cpu().numpy() if parameter.grad is not None else [0])
    for i in range(len(v_n)):
        if np.max(v_v[i]).item() - np.min(v_v[i]).item() < 1e-6:
            color = Bcolors.FAIL + '*'
        else:
            color = Bcolors.OKGREEN + ' '
        print('%svalue %s: %.3e ~ %.3e' % (color, v_n[i], np.min(v_v[i]).item(), np.max(v_v[i]).item()))
        print('%sgrad  %s: %.3e ~ %.3e' % (color, v_n[i], np.min(v_g[i]).item(), np.max(v_g[i]).item()))


def get_dataloader(config: BaseConfig):
    transform = transforms.Compose([transforms.RandomResizedCrop(config.img_size), transforms.ToTensor()])
    if config.name == 'cifar':
        trainset = CIFAR10(root=config.data_dir, train=True, transform=transform)
        train_size = int(0.8 * len(trainset))
        validation_size = len(trainset) - train_size
        trainset, validset = random_split(trainset, [train_size, validation_size])
    elif config.name == 'open':
        trainset = ImgDataset(config.data_dir, transform=transform, image_set='train_f')
        validset = ImgDataset(config.data_dir, transform=transform, image_set='validation')
        train_size = int(0.1 * len(trainset))  # 10万 * 0.1
        trainset, _ = random_split(trainset, [train_size, len(trainset) - train_size])
    elif config.name == 'div':
        trainset = ImgDataset(config.train_dir, transform=transform, image_set='DIV2K_train_HR')
        validset = ImgDataset(config.test_dir, transform=transform, )
    else:
        raise ValueError('not found config')

    train_loader = DataLoader(trainset, batch_size=config.batch_size, pin_memory=True, shuffle=True, num_workers=config.num_workers,
                              drop_last=True)
    val_loader = DataLoader(validset, batch_size=config.batch_size, pin_memory=True, shuffle=False, num_workers=config.num_workers,
                            drop_last=True)
    return train_loader, val_loader


def save_checkpoint(state, filepath, del_previous=False, del_str_start=None):
    if del_previous and del_str_start:
        del_filelist(os.path.dirname(filepath), del_str_start)
    torch.save(state, filepath)


def del_filelist(pardir_path, file_starts):
    for file in Path(pardir_path).iterdir():
        if file.is_file() and file.name.startswith(file_starts):
            file.unlink()


class Timer:
    def __init__(self):
        self._start = 0
        self._end = 0

    def start(self):
        self._start = time.time()

    def end(self):
        self._end = time.time()
        return round((self._end - self._start) * 1000, 2)


class AverageMeter:
    """指标计数器"""

    def __init__(self):
        self._val = 0.
        self._avg = 0.
        self._sum = 0.
        self._count = 0.

    @property
    def val(self):
        """最新记录的值"""
        return round(self._val, 6)

    @property
    def avg(self):
        """均值，精确 6 位"""
        return round(self._avg, 6)

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()
        self._val = val
        self._sum += val * n
        self._count += n
        self._avg = self._sum / self._count

    def clear(self):
        self._val = 0.
        self._avg = 0.
        self._sum = 0.
        self._count = 0.


class FeatureHook():
    def __init__(self, module):
        module.register_forward_hook(self.forward_attach)
        module.register_backward_hook(self.backward_attach)

    def forward_attach(self, module, input, output):
        self.forward_feature = output.cpu()

    def backward_attach(self, module, grad_in, grad_out):
        self.backward_grad = grad_out.cpu()


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook()

    def hook(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_image, target_class):
        output = self.model(input_image)
        self.model.zero_grad()
        target = output[0][target_class]
        target.backward()

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        import cv2
        cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))

        return cam
