#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/3/12 23:33
# @Author  : sy
# @File    : BaseTrainer.py
import copy
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

import torch
from thop import clever_format, profile
from torch import nn
from tqdm import tqdm

from .util import plot_log

# 日志使用方法：https://zhuanlan.zhihu.com/p/166671955
logging.basicConfig(
    format="\033[1;36m %(pathname)s:%(lineno)d:[%(asctime)s] {%(funcName)s} (%(levelname)s): %(message)s\n \033[0m",
    level=logging.INFO,
    datefmt="%m/%d %H:%M:%S"
)

logging.info('BaseTrainer.py begin...')
# 重写print，格式化输出
printf = logging.info


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

    def __init__(self, logger_dir: Union[Path, str] = None, version=0):
        super().__init__()
        self._log_data: Dict[str, List] = {}
        self.log_dir = logger_dir
        self.version = version

        if logger_dir is not None:
            (log_dir := Path(Path.cwd(), logger_dir)).mkdir(parents=True, exist_ok=True)
            num = max(
                (int(name[1]) for path in log_dir.iterdir() if path.is_dir() and len((name := path.name.split("version_"))) > 1),
                default=-1
            )
            self.version = num + 1
            (log_dir / f"version_{self.version}").mkdir()
            self.log_dir = (log_dir / f"version_{self.version}").__str__()

    def get_data(self, key=None):
        return self._log_data[key] if key is not None else self._log_data

    def add_data(self, key=None, value=None, **kwargs):
        for k, v in kwargs.items():
            self._log_data[k] = v
        if key is None:
            return
        if key not in self._log_data:
            self._log_data[key] = []
        self._log_data[key].append(value)

    def save_data(self, file_name="data.json", **kwargs):
        if not self.log_dir:
            raise ValueError('logger_dir is not define')
        self._log_data.update(**kwargs)
        return Path(self.log_dir, file_name).write_text(json.dumps(self._log_data))

    def plot_data(self, plot_key=None, file_name="", *args, **kwargs):
        if not self.log_dir:
            raise ValueError('logger_dir is not define')
        if plot_key is not None:
            kwargs.update({plot_key: self.get_data(plot_key)})
        plot_log(Path(self.log_dir, file_name), *args, **kwargs)


class BaseTrainer(TrainerHook, TrainerLogger):
    def __init__(self, model: nn.Module, logger_dir: Union[Path, str] = None, resume_ckpt: Union[Path, str] = None):
        """小知识：super() 是按照 BaseTrainer.mro() 返回的顺序查找方法，
        顺序是：BaseTrainer -> TrainerHook -> TrainerLogger，即继承类从左至右的顺序
        如果查找到调用的方法名，就会直接执行这个类的这个方法，而**不管参数是否对应的上**
        例如这里的 BaseTrainer 继承了 TrainerHook 和 TrainerLogger，调用 super().__init__(logger_dir)，
        因为 TrainerHook 没有 __init__ 方法，所以会查找到 TrainerLogger；
        但是如果 TrainerHook 有 __init__ 方法，且无参数，则会因为**参数不匹配报错**
        """
        super().__init__(logger_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.val_check_interval = 1
        self.current_epoch = 0
        if resume_ckpt is not None:
            self.resume_ckpt = str(resume_ckpt)

    def train(self, train_loader, val_loader, epochs, *args, **kwargs):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.before_train(*args, **kwargs)
        if hasattr(self, 'resume_ckpt'):
            self.load_resume_ckpt(self.resume_ckpt)
        for epoch in range(self.current_epoch, self.epochs):
            self.before_epoch(*args, **kwargs)
            # tqdm使用：https://blog.csdn.net/qq_41554005/article/details/117297861
            self.current_epoch = epoch
            self.pbar = tqdm(
                total=len(self.train_loader),
                desc=f'Epoch {self.current_epoch + 1}/{self.epochs}',
                mininterval=0.3,
                position=0,
                dynamic_ncols=True,
                bar_format="{desc}{percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt}[{elapsed}{postfix},eta={remaining}]"
            )
            self.before_train_one_epoch(*args, **kwargs)
            self.train_one_epoch(*args, **kwargs)
            self.end_train_one_epoch(*args, **kwargs)
            self.pbar.close()
            if epoch % self.val_check_interval == 0:
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
            self.pbar.total_eta = self.pbar.format_interval(((total / rate) * (self.epochs - (self.current_epoch + 1))) + (
                    (total - n) / rate) if rate and total else 0)

            train_batch = self._batch_to_cuda(train_batch)
            self.before_train_one_step(*args, **kwargs)
            self.train_one_step(train_idx, train_batch, *args, **kwargs)
            self.end_train_one_step(*args, **kwargs)
            self.pbar.update(1)

    def train_one_step(self, batch_idx, batch, *args, **kwargs):
        raise NotImplementedError()

    @torch.no_grad()
    def val_one_epoch(self, *args, **kwargs):
        self.model.eval()
        for val_idx, val_batch in enumerate(self.val_loader):
            val_batch = self._batch_to_cuda(val_batch)
            self.before_val_one_step(*args, **kwargs)
            self.val_one_step(val_idx, val_batch, *args, **kwargs)
            self.end_val_one_step(*args, **kwargs)

    @torch.no_grad()
    def val_one_step(self, batch_idx, batch, *args, **kwargs):
        raise NotImplementedError()

    @torch.no_grad()
    def test_one_epoch(self, *args, **kwargs):
        self.model.eval()
        for test_idx, test_batch in enumerate(self.test_loader):
            test_batch = self._batch_to_cuda(test_batch)
            self.before_test_one_step(*args, **kwargs)
            self.test_one_step(test_idx, test_batch, *args, **kwargs)
            self.end_test_one_step(*args, **kwargs)

    @torch.no_grad()
    def test_one_step(self, batch_idx, batch, *args, **kwargs):
        raise NotImplementedError()

    def load_ckpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        ckpt = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        ckpt = {k: v for k, v in ckpt.items() if not (('total_ops' in k) or ('total_params' in k))}
        self.model.load_state_dict(ckpt, True)

    def load_resume_ckpt(self, ckpt_path, *args, **kwargs):
        pass

    def save_parameters(self, params, **kwargs):
        if not isinstance(params, dict):
            raise ValueError("params must be a dict")
        if not self.log_dir:
            raise ValueError("logger_dir is not define")
        params.update({'datetime': datetime.now().strftime('%Y/%m/%d %H:%M:%S')})
        if hasattr(self, 'resume_ckpt'):
            params.update({'resume_ckpt': self.resume_ckpt})
        params.update(**kwargs)
        with open(os.path.join(self.log_dir, 'parameters.json'), 'w') as f:
            json.dump(params, f)

    @torch.no_grad()
    def compute_cost(self, *forward_inputs):
        net = copy.deepcopy(self.model)
        forward_inputs = [input.to(self.device) for input in forward_inputs if isinstance(input, torch.Tensor)]
        # profile 会向模型中添加额外的参数，因此需要使用 copy.deepcopy，避免最后保存模型时有不必要的参数
        flops, params = clever_format(profile(net, inputs=forward_inputs, verbose=False), "%.3f")
        print('FLOPs = ' + flops)
        print('Params = ' + params)

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

    def _batch_to_cuda(self, batch):
        """遍历数据，将所有 cpu 上的 Tensor 转移到指定设备"""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, list):
            return [self._batch_to_cuda(b) for b in batch]
        elif isinstance(batch, tuple):
            return tuple(self._batch_to_cuda(b) for b in batch)
        elif isinstance(batch, set):
            return {self._batch_to_cuda(b) for b in batch}
        elif isinstance(batch, dict):
            return {key: self._batch_to_cuda(value) for key, value in batch.items()}
        else:
            return batch
