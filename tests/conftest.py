"""共享 fixtures：合成数据目录，替代 gitignore 的真实 img/coco 数据（hermetic 测试）。

真实 `img/` 目录不随 git/worktree/CI 走，测试若硬依赖它必然在其他环境失败。
本 fixture 在 tmp_path 里按 HBSNDataset 的读取格式（{i}.png + {i}.npy）生成
最小合成样本，让数据集测试在任何环境都可复现。
"""
import numpy as np
import pytest
from PIL import Image

SAMPLE_SIZE = 256  # 与 HbsnDatasetSchema 默认 height/width 一致


def _write_sample(dirpath, idx):
    """写 {idx}.png（256×256 灰度）+ {idx}.npy（float32 CHW (2,256,256)）。"""
    rng = np.random.default_rng(idx)
    img = (rng.random((SAMPLE_SIZE, SAMPLE_SIZE)) * 255).astype(np.uint8)
    Image.fromarray(img, mode="L").save(f"{dirpath}/{idx}.png")
    hbs = rng.random((2, SAMPLE_SIZE, SAMPLE_SIZE), dtype=np.float32)
    hbs[1] += 1.0  # 保证 std > 0（test_hbsn_dataset_npy_values 断言）
    np.save(f"{dirpath}/{idx}.npy", hbs)


@pytest.fixture
def hbsn_data_dir(tmp_path):
    """生成 train/test 两个合成数据目录，返回 (train_dir, test_dir)。"""
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    test_dir.mkdir()
    for i in range(3):
        _write_sample(str(train_dir), i)
    _write_sample(str(test_dir), 0)
    return str(train_dir), str(test_dir)
