"""测试形状边界点生成（单位圆盘内单连通 Jordan 域，C^1 边界）。

统一约定：边界点为 (n,2) float 数组、逆时针、**无重复端点**（linspace endpoint=False，
否则 hbs 库 zipper 会折叠——见 _compat.py 说明）。
"""

import numpy as np


def closed_curve(r_func, n=300):
    """半径函数 r(θ) → (n,2) 边界点（逆时针，无重复端点）。"""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = r_func(t)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def circle(n=300, r=0.85):
    return closed_curve(lambda t: np.full_like(t, r), n)


def ellipse(n=300, a=1.0, b=0.7):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([a * np.cos(t), b * np.sin(t)], axis=1)


def star(n=300, k=5, depth=0.15, base=0.9):
    """k 叶半径微扰（覆盖旋转对称：k 越大越接近 Σ，k=1 即偏心圆）。"""
    return closed_curve(lambda t: base + depth * np.cos(k * t), n)


def random_polygon(n=300, verts=6, radius=0.85, rng=None):
    """随机多边形（凸包近似，避免自交）：随机顶点 → 角度排序。

    rng 传入可复现（默认无种子——调用方应传固定 seed 的 rng，否则每次不同）。
    """
    if rng is None:
        rng = np.random.default_rng()
    z = (
        radius
        * (1 + 0.4 * rng.random(verts))
        * np.exp(1j * rng.uniform(0, 2 * np.pi, verts))
    )
    idx = np.argsort(np.angle(z))
    z = z[idx]
    t = np.linspace(0, 1, n, endpoint=False)
    # 顶点间线性插值 → 闭合多边形边界
    zz = []
    for i in range(verts):
        a, b = z[i], z[(i + 1) % verts]
        seg = a + (b - a) * t
        zz.append(seg[:-1])
    zz = np.concatenate(zz)
    zz = np.concatenate([zz, [z[0]]])
    return np.stack([zz.real, zz.imag], axis=1)


def quadrilateral(n=300):
    """固定不规则四边形（非对称、凸）——E4 不变性代表形状（可复现）。

    顶点在单位圆内，边界线性插值后质心归零 + 缩放 0.85（与 triangle 同约定）。
    """
    verts = np.array([[0.85, 0.45], [-0.6, 0.75], [-0.9, -0.35], [0.55, -0.8]])
    edges = []
    for i in range(4):
        p0, p1 = verts[i], verts[(i + 1) % 4]
        seg = np.linspace(0, 1, n // 4 + 1, endpoint=False)
        edges.append(p0 + (p1 - p0) * seg[:, None])
    bd = np.concatenate(edges)
    bd = bd - bd.mean(axis=0)
    bd = bd / np.abs(bd).max() * 0.85
    return bd


def triangle(t, n=300, base=0.7):
    """E1 三角形单参数族：两顶点固定，第三顶点在 (cos(πt), sin(πt)) 扫过。

    t∈[0,1]，t=0.5 时三角形过等腰构型（I₂(B_t) 穿越实轴处）。
    顶点：A=(-b,0), B=(b,0) 固定；C(t)=(cos(πt), sin(πt)) 在上半平面扫过。
    """
    b = base
    a = np.array([[-b, 0.0], [b, 0.0], [np.cos(np.pi * t), np.sin(np.pi * t)]])
    # 线性插值三条边 → 300 边界点
    edges = []
    for i in range(3):
        p0, p1 = a[i], a[(i + 1) % 3]
        seg = np.linspace(0, 1, n // 3 + 1, endpoint=False)
        edges.append(p0 + (p1 - p0) * seg[:, None])
    bd = np.concatenate(edges)
    # 移到单位圆盘内（质心到原点，尺度归一）
    bd = bd - bd.mean(axis=0)
    bd = bd / np.abs(bd).max() * 0.85
    return bd


def triangle_slide(x, n=300, base=0.7, h=0.9):
    """E1 滑动顶点族：底边两顶点固定，上顶点 C(x)=(x,h) 沿 x 方向平移。

    A=(-base,0), B=(base,0) 固定；C(x)=(x, h) 在上方水平滑动。
    x=0 时等腰（y 轴对称 → I₂ 纯虚数，非翻转点）；翻转发生在 Im I₂ 穿越实轴处，
    由镜像对称成 ±x* 一对。h 对齐旧半圆扫掠族翻转处顶点高度。
    """
    a = np.array([[-base, 0.0], [base, 0.0], [x, h]])
    edges = []
    for i in range(3):
        p0, p1 = a[i], a[(i + 1) % 3]
        seg = np.linspace(0, 1, n // 3 + 1, endpoint=False)
        edges.append(p0 + (p1 - p0) * seg[:, None])
    bd = np.concatenate(edges)
    bd = bd - bd.mean(axis=0)
    bd = bd / np.abs(bd).max() * 0.85
    return bd


def shape_to_image(bound, h=256, w=256):
    """边界点 → 二值形状图（黑底白形状），供 HBSN 输入 / 噪声实验。"""
    import matplotlib

    matplotlib.use("Agg")  # 无头环境，避免 tostring_rgb 依赖 GUI
    import matplotlib.pyplot as plt

    z = bound[:, 0] + 1j * bound[:, 1]
    z = z - z.mean()
    z = z / np.abs(z).max() * 0.85
    fig = plt.figure(figsize=(w, h), facecolor="black", dpi=1)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.axis("off")
    plt.fill(z.real, z.imag, color="white")
    plt.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()  # (h,w,3) RGB
    plt.close(fig)
    return img


def boundary_from_image(img_gray):
    """二值形状图（黑底白）→ 边界点 (n,2)。E2 噪声图用；hbs 库 get_boundary 对黑底白有 bug。"""
    import cv2

    binary = (img_gray > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    main = max(contours, key=cv2.contourArea)
    pts = main.reshape(-1, 2).astype(np.float64)
    return pts
