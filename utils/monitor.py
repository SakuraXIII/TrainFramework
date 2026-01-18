#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026/1/18 15:04
# @Author  : SakuHx
# @File    : monitor.py

import threading
from dataclasses import dataclass, replace, asdict
from enum import Enum
from typing import Dict, Any, Callable, Optional, List

import paddle as pd
from paddle.optimizer import Optimizer

from utils.common import init_logger

logger = init_logger()


# ------------------------------------------------ #


def lr_provider(optimizer: Optimizer) -> Dict[str, Any]:
    return {"learning_rate": optimizer.get_lr() or 0.0}


def grad_norm_provider(model) -> Dict[str, Any]:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return {"grad_norm": total_norm ** 0.5}


def gpu_mem_provider(device) -> Dict[str, Any]:
    if pd.device.get_device() is not 'cpu':
        mem = pd.device.memory_allocated(device) / 1024 ** 2
        return {"gpu_memory_mb": round(mem, 1)}
    return {}


class Status(Enum):
    FAILED = "失败"
    INIT = "初始化中"
    TRAIN = "训练中"
    SUCCESS = "成功"

    def to_json(self):
        return self.value


@dataclass
class TrainState:
    status: Status = Status.INIT
    epoch: int = 0
    total_epoch: int = 0
    batch_idx: int = 0
    learning_rate: float = 0.0
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    # 后续扩展字段需在此声明
    grad_norm: Optional[float] = None
    gpu_memory_mb: Optional[int] = None

    def to_dict(self):
        """将对象转换为字典，枚举转为其值"""
        d = asdict(self)
        d['status'] = self.status.value  # 或者用 self.color.name
        return d

    @classmethod
    def from_dict(cls, data):
        """从字典创建对象，将值转换回枚举"""
        status = data.pop('status')
        # 假设 color_value 是枚举的 value，如果是 name 则用 Color[color_value]
        status_enum = Status(status)
        return cls(**data, status=status_enum)


class TrainMonitor:
    def __init__(self):
        self._state = TrainState()
        self._lock = threading.RLock()
        self._callbacks: List[Callable[[TrainState], None]] = []
        self._providers: List[Callable[[], Dict[str, Any]]] = []

    def update(self, **kwargs):
        with self._lock:
            # replace: 专为 @dataclass 类设计，轻量拷贝（浅拷贝）返回不可变新实例
            self._state = replace(self._state, **kwargs)

    def get_snapshot(self) -> TrainState:
        with self._lock:
            self.refresh_snapshot()
            return replace(self._state)  # immutable copy 不可变拷贝

    def register_provider(self, provider: Callable[[], Dict[str, Any]]):
        """注册一个指标提供者：每次 snapshot 时调用它补全字段"""
        self._providers.append(provider)

    def refresh_snapshot(self):
        """合并所有 provider 数据到快照"""
        data = {}
        for p in self._providers:
            try:
                data.update(p())
            except Exception as e:
                logger.error(f"[Monitor] Provider error: {e}")
        self.update(**data)

    def clear_state(self):
        self._state = TrainState()

    def reset_all(self):
        self._state = TrainState()
        self._callbacks.clear()
        self._providers.clear()

    # 其他钩子：on_epoch_start/end, on_train_start/end...
