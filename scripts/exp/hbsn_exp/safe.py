"""get_hbs 安全调用：信号超时（hbs 库对退化/近对称边界可能无限循环）。

hbs 1.0.0 的两个 while 循环无迭代上限：
- conformal_welding.y_post_norm：Möbius 居中退化边界可能不收敛
- get_hbs 归一化循环：Σhbs≈0（近旋转对称形状）时 angle(≈0) 噪声振荡

纯 Python/NumPy 计算中 SIGALRM 在字节码间隙触发，可打断 CPU 循环。
"""

import signal


class HBSTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise HBSTimeout(f"get_hbs 超过 {getattr(_alarm_handler, '_secs', '?')}s")


def timed_hbs(get_hbs, *args, timeout=15.0, **kwargs):
    """带超时的 get_hbs 调用。超时抛 HBSTimeout，由调用方决定重试/跳过。"""
    _alarm_handler._secs = timeout
    old = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(int(timeout))
    try:
        return get_hbs(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
