from typing import Any, Dict, List


class BaseConfig:
    def __init__(self, config_dict: Dict[str, Any] = {}):
        self.load_config(config_dict)

    def load_config(self, config_dict: Dict[str, Any] = {}):
        for attr in dir(self):
            if attr in config_dict:
                self.__setattr__(attr, config_dict[attr])

    @property
    def _except_keys(self):
        return ["load_config"]

    @property
    def _show_keys(self):
        return [
            attr
            for attr in dir(self)
            if (
                not attr.startswith("_")
                and not attr.startswith("get")
                and attr not in self._except_keys
            )
        ]

    def get_config(self):
        return {
            attr: self.__getattribute__(attr)
            for attr in self._show_keys
        }


class RunConfig(BaseConfig):
    checkpoint_path: str = ""
    version: str = "0.0"

    total_epoches: int = 1000
    batch_size: int = 64

    weight_norm: float = 1e-5
    moments: float = 0.9
    lr: float = 1e-4
    lr_decay_rate: float = 0.5
    lr_decay_steps: list = [50, 100]


class Config:
    """
    Total configuration for the project,
    cataining the configuration of network, dataset, recorder and run.
    """

    def __init__(
        self,
        net_config,
        dataset_config,
        recorder_config,
        run_config,
    ):
        self.net_config = net_config
        self.dataset_config = dataset_config
        self.recorder_config = recorder_config
        self.run_config = run_config

    def load_config(self, config_dict: Dict[str, Any] = {}):
        for config in [
            self.net_config,
            self.dataset_config,
            self.recorder_config,
            self.run_config,
        ]:
            config.load_config(config_dict)

    def get_config(self) -> Dict[str, Any]:
        return {
            k: v
            for config in [
                self.net_config,
                self.dataset_config,
                self.recorder_config,
                self.run_config,
            ]
            for k, v in config.get_config().items()
        }

    def get_config_str_list(self) -> List[str]:
        return [f"{k}: {v}" for k, v in self.get_config().items()]
