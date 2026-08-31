from PyQt5.QtWidgets import (
    QTreeWidget, QTreeWidgetItem, QWidget, QVBoxLayout,
    QLabel, QPushButton, QHeaderView, QMessageBox, QAbstractItemView
)
from PyQt5.QtCore import pyqtSignal, Qt
from pathlib import Path
import os


class FileTreeWidget(QWidget):
    """
    左侧文件树/项目管理面板。
    支持浏览目录结构、显示最近打开的文件列表，双击加载 EIGENVAL。
    """

    # 信号：用户请求加载指定路径的 EIGENVAL 文件
    file_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._recent_files: list = []
        self._current_project_dir: Path = Path('.')
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # --- 最近打开文件 ---
        self.label_recent = QLabel("<b>Recent Files</b>")
        layout.addWidget(self.label_recent)

        self.tree_recent = QTreeWidget()
        self.tree_recent.setHeaderHidden(True)
        self.tree_recent.setMaximumHeight(150)
        self.tree_recent.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree_recent.itemDoubleClicked.connect(self._on_recent_double_clicked)
        layout.addWidget(self.tree_recent)

        self.btn_clear_recent = QPushButton("Clear Recent")
        self.btn_clear_recent.clicked.connect(self.clear_recent)
        layout.addWidget(self.btn_clear_recent)

        # --- 项目目录浏览 ---
        self.label_project = QLabel("<b>Project Browser</b>")
        layout.addWidget(self.label_project)

        self.tree_project = QTreeWidget()
        self.tree_project.setHeaderHidden(True)
        self.tree_project.itemDoubleClicked.connect(self._on_project_double_clicked)
        layout.addWidget(self.tree_project)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh_project_tree)
        layout.addWidget(self.btn_refresh)

        layout.addStretch()

    # ------------------------------------------------------------------
    # 最近文件
    # ------------------------------------------------------------------
    def add_recent_file(self, filepath: str):
        """添加文件到最近列表（去重，置顶）"""
        path = str(Path(filepath).resolve())
        if path in self._recent_files:
            self._recent_files.remove(path)
        self._recent_files.insert(0, path)
        self._recent_files = self._recent_files[:10]  # 最多保留 10 条
        self._refresh_recent_tree()

    def clear_recent(self):
        """清空最近文件列表"""
        self._recent_files.clear()
        self._refresh_recent_tree()

    def set_recent_files(self, files: list):
        """批量设置最近文件列表"""
        self._recent_files = [str(Path(f).resolve()) for f in files]
        self._refresh_recent_tree()

    def get_recent_files(self) -> list:
        return self._recent_files.copy()

    def _refresh_recent_tree(self):
        self.tree_recent.clear()
        for fp in self._recent_files:
            p = Path(fp)
            item = QTreeWidgetItem(self.tree_recent)
            item.setText(0, p.name)
            item.setToolTip(0, str(p))
            item.setData(0, Qt.UserRole, str(p))

    def _on_recent_double_clicked(self, item: QTreeWidgetItem, column: int):
        path = item.data(0, Qt.UserRole)
        if path:
            self._try_load(path)

    # ------------------------------------------------------------------
    # 项目目录浏览
    # ------------------------------------------------------------------
    def set_project_directory(self, directory: str):
        """设置要浏览的项目根目录"""
        d = Path(directory)
        if d.exists() and d.is_dir():
            self._current_project_dir = d
            self.refresh_project_tree()

    def refresh_project_tree(self):
        """刷新项目目录树"""
        self.tree_project.clear()
        if not self._current_project_dir.exists():
            return

        root_item = QTreeWidgetItem(self.tree_project)
        root_item.setText(0, self._current_project_dir.name)
        root_item.setData(0, Qt.UserRole, str(self._current_project_dir))
        self._populate_tree(root_item, self._current_project_dir)
        self.tree_project.expandItem(root_item)

    def _populate_tree(self, parent_item: QTreeWidgetItem, directory: Path):
        """递归填充目录树，只显示感兴趣的目录和文件"""
        try:
            entries = sorted(directory.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            return

        for entry in entries:
            # 跳过隐藏文件和缓存目录
            if entry.name.startswith('.') or entry.name == '__pycache__':
                continue

            item = QTreeWidgetItem(parent_item)
            item.setText(0, entry.name)
            item.setData(0, Qt.UserRole, str(entry))

            if entry.is_dir():
                # 递归填充子目录，但限制深度以提升性能
                depth = self._get_item_depth(item)
                if depth < 3:
                    self._populate_tree(item, entry)
            elif entry.name.upper() == 'EIGENVAL':
                # 高亮标记 EIGENVAL 文件
                item.setForeground(0, Qt.darkGreen)

    def _get_item_depth(self, item: QTreeWidgetItem) -> int:
        depth = 0
        while item.parent():
            depth += 1
            item = item.parent()
        return depth

    def _on_project_double_clicked(self, item: QTreeWidgetItem, column: int):
        path = item.data(0, Qt.UserRole)
        if not path:
            return
        p = Path(path)
        if p.is_file() and p.name.upper() == 'EIGENVAL':
            self._try_load(str(p))
        elif p.is_dir():
            # 展开/折叠目录
            if self.tree_project.isItemExpanded(item):
                self.tree_project.collapseItem(item)
            else:
                self.tree_project.expandItem(item)

    # ------------------------------------------------------------------
    # 公共辅助
    # ------------------------------------------------------------------
    def _try_load(self, path: str):
        """尝试加载指定路径，失败时弹窗提示"""
        p = Path(path)
        if not p.exists():
            QMessageBox.warning(self, "Warning", f"File not found:\n{path}")
            return
        self.file_selected.emit(str(p))

    def scan_for_materials(self, base_dir: str) -> list:
        """
        扫描目录，返回所有包含 EIGENVAL 的 (目录名, eigenval路径) 列表。
        用于自动发现示例数据。
        """
        results = []
        base = Path(base_dir)
        if not base.exists():
            return results

        for root, dirs, files in os.walk(base):
            if 'EIGENVAL' in [f.upper() for f in files]:
                root_path = Path(root)
                eigenval = root_path / 'EIGENVAL'
                results.append((root_path.name, str(eigenval)))
        return results
