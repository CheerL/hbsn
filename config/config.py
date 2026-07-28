from typing import Any, Dict, List

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
