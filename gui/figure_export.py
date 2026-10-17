"""共享的 matplotlib 画布行为。"""


class FigureExportMixin:
    """为持有 ``figure`` 属性的 widget 提供统一的图片导出。

    两个绘图组件此前各写了一份完全相同的 ``save_figure``。
    """

    def save_figure(self, filepath: str, dpi: int = 300):
        """导出高分辨率图片（PNG/SVG 由扩展名决定）。"""
        self.figure.savefig(filepath, dpi=dpi, bbox_inches='tight')
