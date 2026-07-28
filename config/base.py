from typing import Any, Dict

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
        return {attr: self.__getattribute__(attr) for attr in self._show_keys}