"""Recorder：损失记录/checkpoint 判定/结果图（CPU，无 GPU 依赖）。"""
import pytest
import torch

from hbsn.config.schemas import RecorderSchema
from hbsn.recorder import Recorder, get_random_index


def make_recorder(tmp_path, **kw):
    cfg = RecorderSchema(log_dir=str(tmp_path / "run"), comment="test", **kw)
    return Recorder(cfg, train_size=4, test_size=2, total_epoches=2, batch_size=2)


def test_set_log_dir_comment(tmp_path):
    cfg = RecorderSchema(log_base_dir=str(tmp_path), comment="foo")
    r = Recorder(cfg, 1, 1, 1, 1)
    assert "foo" in r.log_dir
    assert r.checkpoint_dir.endswith("checkpoints")


def test_get_random_index():
    idx, n = get_random_index(3, 10)
    assert n == 3 and len(idx) == 3
    idx, n = get_random_index(20, 5)  # num > size → 截断
    assert n == 5 and len(idx) == 5


def test_add_loss_train_test(tmp_path):
    r = make_recorder(tmp_path)
    r.add_loss(0, 0, {"loss": torch.tensor(0.5), "iou": torch.tensor(0.1)}, is_train=True)
    r.add_loss(0, 1, {"loss": torch.tensor(0.4)}, is_train=True)
    r.add_loss(0, 0, {"loss": torch.tensor(0.3)}, is_train=False)
    assert r.train_loss[0].item() == pytest.approx(0.5)
    assert r.train_loss[1].item() == pytest.approx(0.4)
    assert r.train_iou[0].item() == pytest.approx(0.1)
    assert r.test_loss[0].item() == pytest.approx(0.3)


def test_add_epoch_loss(tmp_path):
    r = make_recorder(tmp_path)
    r.train_loss[0] = torch.tensor(0.5)
    r.train_loss[1] = torch.tensor(0.7)
    r.test_loss[0] = torch.tensor(0.4)
    r.test_loss[1] = torch.tensor(0.6)
    r.add_epoch_loss(0, is_train=True)
    r.add_epoch_loss(0, is_train=False)


def test_update_best(tmp_path):
    r = make_recorder(tmp_path)  # test_size=2 → mean 取两个槽
    r.test_loss[0] = torch.tensor(0.5)
    r.test_loss[1] = torch.tensor(0.5)
    assert r.update_best(0) is True
    assert r.best_epoch == 0 and r.best_loss == pytest.approx(0.5)
    r.test_loss[0] = torch.tensor(0.7)
    r.test_loss[1] = torch.tensor(0.7)  # mean 0.7 > 0.5 → 不更新
    assert r.update_best(1) is False
    assert r.best_epoch == 0


def test_init_recorder(tmp_path):
    r = make_recorder(tmp_path)
    r.init_recorder({"a": "1", "b": "2"})  # net=None → 跳过 add_graph


def test_add_output_variants(tmp_path):
    """三种 output_data 结构：hbsn(2)/seg(3)/tpsn(4)，覆盖所有绘图分支。"""
    r = make_recorder(tmp_path)
    img = torch.rand(2, 1, 16, 16)
    gt_mask = torch.rand(2, 1, 16, 16)
    pred_hbs = torch.rand(2, 2, 16, 16)
    gt_hbs = torch.rand(2, 2, 16, 16)
    mapping = torch.rand(2, 2, 16, 16)

    r.add_output(0, 0, (img, gt_mask), (pred_hbs, gt_hbs), is_train=True, num=2)
    r.add_output(0, 1, (img, gt_mask), (gt_mask, pred_hbs, gt_hbs), is_train=True, num=2)
    r.add_output(0, 2, (img, gt_mask), (gt_mask, pred_hbs, gt_hbs, mapping), is_train=True, num=2)


def test_add_output_rgb_image(tmp_path):
    """COCO 型 RGB 输入 (B,3,H,W)。"""
    r = make_recorder(tmp_path)
    img = torch.rand(2, 3, 16, 16)
    gt = torch.rand(2, 1, 16, 16)
    pred_hbs = torch.rand(2, 2, 16, 16)
    gt_hbs = torch.rand(2, 2, 16, 16)
    r.add_output(0, 0, (img, gt), (pred_hbs, gt_hbs), is_train=True, num=2)
