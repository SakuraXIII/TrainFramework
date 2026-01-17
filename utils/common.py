import json
import logging
import math
import os.path
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import paddle
from colorama import init, Fore, Style
from matplotlib import image as mpimg, pyplot as plt

if TYPE_CHECKING:  # TYPE_CHECKING 在运行时为 False，因此不会导入包，避免内存占用。通常用于开发过程中引用包中的类型定义，且运行时又不需要该包
    from paddle import Tensor  # 用于开发时提供类型注解

init(autoreset=True)


class LoggingFormatter(logging.Formatter):
    logger_color = {
        logging.INFO: Fore.CYAN,
        logging.DEBUG: Fore.BLUE,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA
    }
    
    def format(self, record):
        fmt = self.logger_color.get(record.levelno, Fore.RESET)
        fmt_str = fmt + "%(pathname)s:%(lineno)d:[%(asctime)s] {%(funcName)s} (%(levelname)s): %(message)s" + Style.RESET_ALL
        formatter = logging.Formatter(fmt_str, datefmt="%H:%M:%S")
        return formatter.format(record)


@lru_cache  # 缓存logger对象，也可以避免已存在的logger对象被重复增加handler，导致重复打印日志
def init_logger(logger_name='logger'):
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(LoggingFormatter())
    logger = logging.getLogger(logger_name)  # logging包是全局注册，同一个logger_name得到的是全局唯一的同一个对象
    logger.addHandler(handler)  # 同一个logger对象可以被重复增加handler，会导致重复执行handler，例如重复打印
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.info(f"{logger_name} initialized")
    return logger


def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


def make_dirs(*paths):
    for path in paths:
        p = Path(path)
        if p.suffix:  # is_file 无法正确判断不存在的文件路径
            p = p.parent
        p.mkdir(parents=True, exist_ok=True)


def read_json_fromfile(json_file):
    if not os.path.exists(json_file):
        raise FileNotFoundError("json文件不存在: {}".format(json_file))
    with open(json_file, 'r') as f:
        return json.load(f)


def show_img_from_data(data):
    """可视化展示图像，支持Tensor，ndarray，文件路径类型"""
    if isinstance(data, (str, Path)) and os.path.exists(data):
        nd_img = mpimg.imread(data)
    elif isinstance(data, Tensor):
        if data.dim() == 3 and data.shape[0] in (1, 3):
            data = data.transpose([1, 2, 0])  # CHW -> HWC
        nd_img = data.numpy()
    else:  # ndarray
        nd_img = data
    
    cmap = 'gray' if nd_img.ndim == 2 or nd_img.shape[-1] == 1 else None
    plt.imshow(nd_img, cmap=cmap)
    plt.show()


def feature_visualization(x, module_type, stage, n=32, save_dir=Path('runs/detect/exp')):
    """
    x:              Features to be visualized
    module_type:    Module type
    stage:          Module stage within model
    n:              Maximum number of feature maps to plot
    save_dir:       Directory to save results
    """
    if 'Detect' not in module_type:
        batch, channels, height, width = x.shape  # batch, channels, height, width
        if height > 1 and width > 1:
            f = f"stage{stage}_{module_type.split('.')[-1]}_features.png"  # filename
            
            blocks = paddle.chunk(x[0].cpu(), channels, dim=0)  # select batch index 0, block by channels
            n = min(n, channels)  # number of plots
            fig, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 8 rows x n/8 cols
            ax = ax.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            for i in range(n):
                ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
                ax[i].axis('off')
            
            print(f'Saving {save_dir / f}... ({n}/{channels})')
            plt.savefig(save_dir / f, dpi=300, bbox_inches='tight')
            plt.close()


class Timer:
    """计时器"""
    
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
        self.clear()
    
    @property
    def val(self):
        """最新记录的值"""
        return round(self._val, 6)
    
    @property
    def avg(self):
        """均值，精确 6 位"""
        return round(self._avg, 6)
    
    def update(self, val, n=1):
        self._val = val
        self._sum += val * n
        self._count += n
        self._avg = self._sum / self._count
    
    def clear(self):
        self._val = 0.
        self._avg = 0.
        self._sum = 0.
        self._count = 0.
