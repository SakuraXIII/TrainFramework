#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2025/12/29 19:42
# @Author  : sy
# @File    : trainer.py
import copy
import json
import os
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import paddle as pd
from paddle import nn
from paddle.optimizer import Optimizer
from tqdm import tqdm

from config.config import Config
from utils.common import init_logger
from utils.monitor import TrainMonitor, lr_provider, gpu_mem_provider

logger = init_logger()


# ------------------------------------------------ #

class TrainerHook:
    def before_train(self, *args, **kwargs):
        pass

    def before_epoch(self, *args, **kwargs):
        pass

    def before_train_one_epoch(self, *args, **kwargs):
        pass

    def before_train_one_step(self, *args, **kwargs):
        pass

    def before_val_one_epoch(self, *args, **kwargs):
        pass

    def before_val_one_step(self, *args, **kwargs):
        pass

    def end_train_one_epoch(self, *args, **kwargs):
        pass

    def end_train_one_step(self, *args, **kwargs):
        pass

    def end_val_one_epoch(self, *args, **kwargs):
        pass

    def end_val_one_step(self, *args, **kwargs):
        pass

    def end_epoch(self, *args, **kwargs):
        pass

    def end_train(self, *args, **kwargs):
        pass

    def before_test(self, *args, **kwargs):
        pass

    def before_test_one_epoch(self, *args, **kwargs):
        pass

    def before_test_one_step(self, *args, **kwargs):
        pass

    def end_test_one_step(self, *args, **kwargs):
        pass

    def end_test_one_epoch(self, *args, **kwargs):
        pass

    def end_test(self, *args, **kwargs):
        pass


class TrainerLogger:

    def __init__(self, logger_dir: Union[Path, str], version=0, use_vdl=True):
        super().__init__()
        self._log_data: Dict[str, List] = {}
        self.log_dir = logger_dir
        self.version = version or 0
        self.use_vdl = use_vdl
        self.log_writer = None

        (log_dir := Path(Path.cwd(), logger_dir)).mkdir(parents=True, exist_ok=True)
        num = max(
            (int(name[1]) for path in log_dir.iterdir() if path.is_dir()
             and len((name := path.name.split("version_"))) > 1),
            default=-1
        )
        self.version = num + 1
        (log_dir / f"version_{self.version}").mkdir()
        self.log_dir = log_dir / f"version_{self.version}"

    def init_vdl(self):
        if self.use_vdl:
            try:
                from visualdl import LogWriter
                self.log_writer = LogWriter(self.log_dir.__str__())
                logger.info(f"请执行命令以访问可视化页面：`visualdl --logdir {self.log_dir.parent} -p 8080`")
                # 等效命令行命令：`visualdl --logdir logdir`
                # from visualdl.server.app import run as vdl
                # vdl(self.log_dir.parent, host="127.0.0.1", port=port, open_browser=open_browser)
                # logger.info(f"已启动VisualDL可视化，请访问：`http://localhost:8080/`")
            except ImportError:
                logger.warning("跳过vdl：未安装 visualdl，无法可视化训练，请安装：`pip install visualdl`")

    def get_data(self, key=None):
        return self._log_data[key] if key is not None else self._log_data

    def add_image(self, key, img: Union[np.ndarray, pd.Tensor], iter_step=None):
        if img is None:
            raise ValueError("传入数据为None")
        if self.log_writer is not None:
            if isinstance(img, pd.Tensor):
                if img.dim() == 3 and img.shape[0] == 3:
                    img = pd.transpose(img, [1, 2, 0]).numpy()

            self.log_writer.add_image(key, img, iter_step)

    def add_data(self, key, value, iter_step=None):
        """iter_step: 本次为第几次迭代（step or epoch）记录数据"""
        if key is None:
            return
        if key not in self._log_data:
            self._log_data[key] = []
        self._log_data[key].append(value)
        if self.log_writer is not None:
            if iter_step is not None:
                self.log_writer.add_scalar(key, value, iter_step)
            else:
                logger.warning("vdl需指定`iter_step`参数")

    def save_data(self, file_name="data.json"):
        if not self.log_dir:
            raise ValueError('logger_dir is not define')
        Path(self.log_dir, file_name).write_text(json.dumps(self._log_data))


class BaseTrainer(TrainerHook):
    def __init__(self, cfg: Config, model: nn.Layer):
        self.cfg = cfg
        self.logger = None if cfg.log_dir is None \
            else TrainerLogger(cfg.log_dir, version=cfg.version, use_vdl=cfg.use_vdl)
        if cfg.epochs is None:
            raise ValueError("Config 需要指定 epochs 训练轮次")
        self.epochs = cfg.epochs
        self.device = self._check_device_available(cfg.device)
        self.model = model.to(self.device)
        self.current_epoch = -1

        if cfg.resume_ckpt is not None:
            self.resume_ckpt = str(cfg.resume_ckpt)
        if self.logger is not None:
            cfg.save_to_file(self.logger.log_dir / "config.yaml")

    def train(self, optimizer: Optimizer, criterion, train_loader, val_loader=None, monitor: TrainMonitor = None, *args, **kwargs):
        if self.logger is not None and self.logger.use_vdl:
            self.logger.init_vdl()
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader

        if monitor is not None:
            self.monitor = monitor
            self.monitor.register_provider(lambda: lr_provider(optimizer))
            self.monitor.register_provider(lambda: gpu_mem_provider(self.device))
            self.monitor.register_provider(lambda: {"total_epoch": self.epochs})

        if val_loader is not None:
            self.val_loader = val_loader
            self.val_check_interval = 1
        self.current_epoch = 0
        self.before_train(*args, **kwargs)
        if hasattr(self, 'resume_ckpt'):
            self.load_resume_ckpt(self.resume_ckpt)
        for epoch in range(self.current_epoch, self.epochs):
            self.before_epoch(*args, **kwargs)
            self.current_epoch = epoch
            self.monitor.update(epoch=epoch)
            # tqdm使用：https://blog.csdn.net/qq_41554005/article/details/117297861
            self.pbar = tqdm(
                total=len(self.train_loader),
                desc=f'Epoch {self.current_epoch + 1}/{self.epochs}',
                mininterval=0.3,
                position=0,
                dynamic_ncols=True,
                bar_format="{desc}{percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt}[{elapsed}{postfix},eta={remaining}]"
            )  # self.pbar.set_postfix({k1:v1, k2:v2, ...}) 将传入键值对文本填充到bar_format {postfix} 中
            self.before_train_one_epoch(*args, **kwargs)
            self.train_one_epoch(*args, **kwargs)
            self.end_train_one_epoch(*args, **kwargs)
            self.pbar.close()
            if val_loader is not None and epoch % self.val_check_interval == 0:
                self.before_val_one_epoch(*args, **kwargs)
                self.val_one_epoch(*args, **kwargs)
                self.end_val_one_epoch(*args, **kwargs)
            self.end_epoch(*args, **kwargs)
        self.end_train(*args, **kwargs)

    def train_one_epoch(self, *args, **kwargs):
        self.model.train()
        for train_idx, train_batch in enumerate(self.train_loader):
            rate = self.pbar.format_dict['rate']
            total = self.pbar.format_dict['total']
            n = self.pbar.format_dict['n']
            self.pbar.total_eta = self.pbar.format_interval(
                ((total / rate) * (self.epochs - (self.current_epoch + 1))) + (
                        (total - n) / rate) if rate and total else 0)

            train_batch = self._batch_to_device(train_batch)
            self.before_train_one_step(*args, **kwargs)
            self.train_one_step(train_idx, train_batch, *args, **kwargs)
            self.end_train_one_step(*args, **kwargs)
            self.pbar.update(1)

    def train_one_step(self, batch_idx, batch, *args, **kwargs):
        raise NotImplementedError("需要实现`train_one_step`方法在训练迭代中期望执行的操作")

    @pd.no_grad()
    def val_one_epoch(self, *args, **kwargs):
        self.model.eval()
        for val_idx, val_batch in enumerate(self.val_loader):
            val_batch = self._batch_to_device(val_batch)
            self.before_val_one_step(*args, **kwargs)
            self.val_one_step(val_idx, val_batch, *args, **kwargs)
            self.end_val_one_step(*args, **kwargs)

    @pd.no_grad()
    def val_one_step(self, batch_idx, batch, *args, **kwargs):
        raise NotImplementedError("需要实现`val_one_step`方法在验证迭代中期望执行的操作")

    @pd.no_grad()
    def test_one_epoch(self, *args, **kwargs):
        self.model.eval()
        for test_idx, test_batch in enumerate(self.test_loader):
            test_batch = self._batch_to_device(test_batch)
            self.before_test_one_step(*args, **kwargs)
            self.test_one_step(test_idx, test_batch, *args, **kwargs)
            self.end_test_one_step(*args, **kwargs)

    @pd.no_grad()
    def test_one_step(self, batch_idx, batch, *args, **kwargs):
        raise NotImplementedError("需要实现`test_one_step`方法在测试迭代中期望执行的操作")

    def load_ckpt(self, ckpt_path):
        if os.path.exists(ckpt_path):
            ckpt = pd.load(ckpt_path)
            ckpt = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            ckpt = {k: v for k, v in ckpt.items() if not (('total_ops' in k) or ('total_params' in k))}
            self.model.load_state_dict(ckpt, True)
        else:
            raise FileNotFoundError("未找到训练检查点文件: {}".format(ckpt_path))

    def load_resume_ckpt(self, ckpt_path, *args, **kwargs):
        pass

    def save_model(self, **kwargs):
        pd.save()

    def compute_cost(self, inputs_shape: list):
        pd.flops(copy.deepcopy(self.model), inputs_shape, print_detail=True)

    def test(self, test_loader, epochs=1, *args, **kwargs):
        self.test_loader = test_loader
        self.epochs = epochs
        self.before_test(*args, **kwargs)
        for epoch in range(0, self.epochs):
            self.before_epoch(*args, **kwargs)
            # tqdm使用：https://blog.csdn.net/qq_41554005/article/details/117297861
            self.current_epoch = epoch
            self.pbar = tqdm(
                total=len(test_loader),
                desc=f'Epoch {self.current_epoch + 1}/{self.epochs}',
                mininterval=0.3,
                position=0,
                dynamic_ncols=True,
                bar_format="{desc}{percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt}[{elapsed}{postfix},eta={remaining}]"
            )
            self.before_test_one_epoch(*args, **kwargs)
            self.test_one_epoch(*args, **kwargs)
            self.end_test_one_epoch(*args, **kwargs)
            self.pbar.close()
            self.end_epoch(*args, **kwargs)
        self.end_test(*args, **kwargs)

    def _batch_to_device(self, batch):
        """遍历数据，将所有 cpu 上的 Tensor 转移到指定设备"""
        if isinstance(batch, pd.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, list):
            return [self._batch_to_device(b) for b in batch]
        elif isinstance(batch, tuple):
            return tuple(self._batch_to_device(b) for b in batch)
        elif isinstance(batch, set):
            return {self._batch_to_device(b) for b in batch}
        elif isinstance(batch, dict):
            return {key: self._batch_to_device(value) for key, value in batch.items()}
        else:
            return batch

    @staticmethod
    def _check_device_available(device):
        final_device = 'cpu'
        if device is None or len(device) == 0:
            logger.warning("未指定设备，默认使用{}".format(final_device))
        elif device in pd.device.get_available_device() or device == 'cpu':
            final_device = device
            logger.info("使用{}设备".format(final_device))
        else:
            logger.warning("{} 不可用，使用{}".format(device, final_device))
        return pd.set_device(final_device)
