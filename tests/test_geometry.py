"""geodesic welding 冒烟：有限值、形状正确。"""
import numpy as np

from hbsn.geometry.geodesic_welding import geodesicwelding


def test_geodesicwelding_finite():
    n = 200
    x = np.arange(n)
    x = (x - x[0]) / (x[-1] - x[-2] + x[-1] - x[0])
    x = np.exp(1j * 2 * np.pi * x)
    rng = np.random.default_rng(0)
    y = x + rng.normal(0, 0.1, n) * 1j
    # 旧生成器用法 geodesicwelding(y, [], y, x)：a 为焊接边界（len n），b 恒为空
    a, b = geodesicwelding(y, [], y, x)
    assert np.isfinite(a).all()
    assert a.shape == (n,)
    assert b.shape == (0,)
