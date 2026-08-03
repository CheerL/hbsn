"""旧 checkpoint 反序列化桩（仅供 tools/migrate_checkpoints.py 使用）。

旧 checkpoint 的 `config` 键是 pickle 的旧 Config 类对象（类路径如
config.net.HBSNetConfig）。torch.load 用 torch 自己的 Unpickler，无法自定义，
所以挂 sys.modules 桩：把旧模块路径映射到这些纯占位类上。占位类属性名与
旧 Config 类逐字一致，unpickle 走 __new__ + __dict__ 恢复，属性对得上即成功。

注意：不要在这里引用 hbsn.config.schemas —— 迁移工具只读旧对象的属性值，
与新版 schema 完全解耦。
"""
import sys
import types


class _Ignore:
    """占位：pickle 中永远不会直接实例化（无 __reduce__ 技巧）。"""


class BaseNetConfigShim:
    pass


class HBSNetConfigShim(BaseNetConfigShim):
    pass


class SegHBSNNetConfigShim(BaseNetConfigShim):
    pass


class MaskRCNNConfigShim(SegHBSNNetConfigShim):
    pass


class TPSNConfigShim(SegHBSNNetConfigShim):
    pass


class BaseDatasetConfigShim:
    pass


class HBSNDatasetConfigShim(BaseDatasetConfigShim):
    pass


class CocoDatasetConfigShim(BaseDatasetConfigShim):
    pass


class RecorderConfigShim:
    pass


class RunConfigShim:
    pass


class ConfigShim:
    pass


# 旧模块路径 → {类名: 占位类}（注意旧拼写 recoder）
LEGACY_MODULES: dict[str, dict[str, type]] = {
    "config": {
        "Config": ConfigShim,
        "BaseConfig": _Ignore,
        "SegNetConfig": SegHBSNNetConfigShim,
        # 旧 config/__init__.py 的再导出路径（pickle 可能按此存）
        "BaseNetConfig": BaseNetConfigShim,
        "HBSNetConfig": HBSNetConfigShim,
        "SegHBSNNetConfig": SegHBSNNetConfigShim,
        "MaskRCNNConfig": MaskRCNNConfigShim,
        "TPSNConfig": TPSNConfigShim,
        "BaseDatasetConfig": BaseDatasetConfigShim,
        "HBSNDatasetConfig": HBSNDatasetConfigShim,
        "CocoDatasetConfig": CocoDatasetConfigShim,
        "RecorderConfig": RecorderConfigShim,
        "RunConfig": RunConfigShim,
    },
    "config.config": {"Config": ConfigShim},
    "config.base": {"BaseConfig": _Ignore},
    "config.net": {
        "BaseNetConfig": BaseNetConfigShim,
        "HBSNetConfig": HBSNetConfigShim,
        "SegHBSNNetConfig": SegHBSNNetConfigShim,
        "MaskRCNNConfig": MaskRCNNConfigShim,
        "TPSNConfig": TPSNConfigShim,
    },
    "config.dataset": {
        "BaseDatasetConfig": BaseDatasetConfigShim,
        "HBSNDatasetConfig": HBSNDatasetConfigShim,
        "CocoDatasetConfig": CocoDatasetConfigShim,
    },
    "config.recoder": {"RecorderConfig": RecorderConfigShim},
    "config.run": {"RunConfig": RunConfigShim},
    # 更早布局（Jun11 时代）的 config 类定义在 net/*.py 与 data/*.py 里，pickle 存的是原始路径
    "net.hbsn": {"HBSNetConfig": HBSNetConfigShim},
    "net.seg_hbsn_net": {"SegHBSNNetConfig": SegHBSNNetConfigShim},
    "net.maskrcnn": {"MaskRCNNConfig": MaskRCNNConfigShim},
    "net.tpsn": {"TPSNConfig": TPSNConfigShim},
    "data.hbsn_dataset": {"HBSNDatasetConfig": HBSNDatasetConfigShim},
    "data.coco_dataset": {"CocoDatasetConfig": CocoDatasetConfigShim},
    "data.base": {"BaseDatasetConfig": BaseDatasetConfigShim},
    # 旧 checkpoint 的 pickle 可能直接引用顶层模块对象（无属性访问），空桩即可
    "net": {},
    "data": {},
    "recorder": {"RecorderConfig": RecorderConfigShim},
    "utils": {},
}


def install() -> None:
    """在 torch.load 之前调用。"""
    for module_name, names in LEGACY_MODULES.items():
        module = types.ModuleType(module_name)
        if not names:
            module.__path__ = []  # 空桩标记为包，允许子模块 import 走 sys.modules
        for class_name, cls in names.items():
            setattr(module, class_name, cls)
        sys.modules[module_name] = module
