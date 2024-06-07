from typing import Any, Dict
from itertools import chain


class BaseConfig:
    def __init__(self, config_dict: Dict[str, Any] = {}):
        self.load_config(config_dict)
    
    def load_config(self, config_dict: Dict[str, Any] = {}):
        for attr in dir(self):
            if attr in config_dict:
                self.__setattr__(attr, config_dict[attr])
 
    @property
    def _except_keys(self):
        return ['load_config']
    
    @property
    def _show_keys(self):
        return [
            attr for attr in dir(self)
            if (
                not attr.startswith('_') and 
                not attr.startswith('get') and 
                attr not in self._except_keys
            )
        ]
        
    def get_config(self):
        return {
            attr: self.__getattribute__(attr)
            for attr in self._show_keys
        }


class RunConfig(BaseConfig):
    load: str = ""
    version: str = "0.0"

    total_epoches: int = 1000
    batch_size: int = 64

    weight_norm: float = 1e-5
    moments: float = 0.9
    lr: float = 1e-3
    lr_decay_rate: float = 0.5
    lr_decay_steps: list = [50, 100]


class Config:
    def __init__(
        self,
        net_config,
        dateset_config,
        recorder_config,
        run_config,
        # config_dict: Dict[str, Any] = {}
    ):
        self.net_config = net_config
        self.dateset_config = dateset_config
        self.recorder_config = recorder_config
        self.run_config = run_config

    def get_config_str_list(self):
        return [
            f"{k}: {v}"
            for k, v in chain(
                [
                    config.get_config().items()
                    for config in [
                        self.net_config,
                        self.dateset_config,
                        self.recorder_config,
                        self.run_config,
                    ]
                ]
            )
        ]