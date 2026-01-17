import os.path
from dataclasses import dataclass

import yaml

from utils.common import make_dirs


@dataclass
class ConfigUtil:
    """配置类的辅助工具实现，一般不需要管"""
    
    @classmethod
    def load_from_file(cls, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError("未找到配置文件: {}".format(file_path))
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data: dict = yaml.safe_load(f) or None
        
        if data is None:
            return cls()
        
        params = {k: v for k, v in data.items() if k in cls.__dict__}
        config = cls(**params)
        
        _ = {setattr(config, k, v) for k, v in data.items() if k not in cls.__dict__}
        
        return config
    
    def save_to_file(self, file_path):
        make_dirs(file_path)
        with open(file_path, 'w', encoding='utf8') as f:
            # __dict__ 只能获取**实例对象**的属性
            # @dataclass 修饰的类中，只有**带类型注解**的属性才会转换为实例属性
            # 因此，想要将属性配置保存到文件中，需要给该属性加上类型注解
            yaml.dump(self.__dict__, f, allow_unicode=True, sort_keys=False, indent=2, default_flow_style=False)
    
    def __repr__(self):
        args = ", ".join([f"{k}={v}" for k, v in self.__dict__.items()])
        return f"{self.__class__.__name__}({args})"


@dataclass
class Config(ConfigUtil):
    device: str = 'cpu'
    num_workers: int = 0
    batch_size: int = 1
    epochs: int = None
    early_stop_epoch: int = None
    
    num_classes: int = 0
    lr: float = None
    
    resume_ckpt: str = None
    log_dir: str = "output"
    version: int = None
    use_vdl: bool = True
    
    dataset_root: str = ""
    
    """
    dataclass 修饰的类中**可变类型**的变量需要通过`dataclasses.field`方法初始化，
    否则该变量在实例对象间共享值
    """


if __name__ == '__main__':
    pass
    Config().save_to_file("./test/config.yaml")
    config = Config.load_from_file("./test/config.yaml")
    c = str(config)
    print(c)
