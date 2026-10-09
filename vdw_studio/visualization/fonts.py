"""matplotlib 中文字体配置。

Windows（微软雅黑/黑体）与 Linux（Noto/WenQuanYi）常见中文字体的
自动探测；树莓派（Bookworm 自带 Noto Sans CJK）同样适用。
调用 :func:`setup_cjk_fonts` 后再创建图表即可正常显示中文；
返回值指示是否找到可用字体，调用方可据此回退英文标注。
"""

from __future__ import annotations

import matplotlib
import re
from matplotlib import font_manager

_CJK_FONTS = [
    "Microsoft YaHei",       # Windows
    "SimHei",                # Windows
    "PingFang SC",           # macOS
    "Noto Sans CJK SC",      # Linux / 树莓派 Bookworm
    "Noto Sans CJK JP",
    "WenQuanYi Zen Hei",     # Linux 常见
    "WenQuanYi Micro Hei",
    "Droid Sans Fallback",   # Android 系 / 部分精简发行版
]


def setup_cjk_fonts() -> bool:
    """配置 matplotlib 使用可用中文字体。返回是否成功。

    成功后中文标题/轴标签正常渲染；同时关闭负号替换
    （``axes.unicode_minus=False``，中文字体常缺 U+2212）。
    """
    installed = {f.name for f in font_manager.fontManager.ttflist}
    found = [f for f in _CJK_FONTS if f in installed]
    if not found:
        return False
    matplotlib.rcParams["font.sans-serif"] = list(dict.fromkeys(found +
        matplotlib.rcParams["font.sans-serif"]))
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["axes.unicode_minus"] = False
    return True


def plot_text(text, fallback):
    """Keep requested CJK text only with an available font; otherwise use English."""
    unsupported = r'[\u3000-\u303f\u3400-\u9fff\uff00-\uffef]'
    if not re.search(unsupported, text) or setup_cjk_fonts():
        return text
    return re.sub(unsupported, '', fallback).strip()
