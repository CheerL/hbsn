from config.base import BaseConfig

class RecorderConfig(BaseConfig):
    """
    If `log_dir` is set, the log will be saved in the `log_dir`
    and the `comment` and `log_base_dir` will be ignored.

    Otherwise, the log will be saved in
    `{$log_base_dir}\{$current_time}_{$comment}`.

    The second way is recommended.
    """

    log_dir = ""
    log_base_dir = "runs"
    comment = ""
    is_add_graph = True

    @property
    def _except_keys(self):
        # only show `log_dir` in the config
        return super()._except_keys + ["log_base_dir", "comment"]