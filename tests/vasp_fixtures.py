"""共享的 VASP 文本夹具。

此前 ``write_eigenval`` / ``write_poscar`` 定义在 ``test_bandviz_mass`` 与
``test_bandviz_parser`` 里，其余测试模块通过 ``from test_bandviz_mass
import ...`` 互相引用——依赖 pytest 的 sys.path 注入，也让测试文件之间
形成隐藏耦合。统一放在这里，``test_*`` 文件只做兼容性再导出。
"""

import numpy as np

# 六方晶格夹具（Å）：用于验证标量/负体积 POSCAR 的距离标定
RAW = np.array([[2., 0, 0], [-1, np.sqrt(3), 0], [0, 0, 12.]])


def write_poscar(path_dir, scale="1", matrix=RAW):
    path = path_dir / "POSCAR"
    path.write_text("Lattice fixture\n" + scale + "\n" +
                    "\n".join(" ".join(map(str, row)) for row in matrix) + "\n", encoding="utf-8")
    return path


def write_eigenval(path_dir, points, energies=None):
    """两带物理回归夹具：可自定义逐 k 点的 (价带, 导带) 能量。"""
    lines = ["1 1 1 1", "1 1 1 1 0", "0", "CAR", "Physical regression", f"2 {len(points)} 2"]
    for i, point in enumerate(points):
        ev, ec = (-1, 2) if energies is None else energies[i]
        lines.extend(["", " ".join(map(str, point)) + " 1", f"1 {ev} 2", f"2 {ec} 0"])
    path = path_dir / "EIGENVAL"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_eigenval_simple(path_dir, points=None, ispin=1, nelect=2):
    """解析器回归夹具：固定两带能量，覆盖 ISPIN=1/2 两套列布局。"""
    if points is None:
        points = [[0, 0, 0], [.25, 0, 0], [.5, 0, 0]]
    lines = [f"1 1 1 {ispin}", "1 1 1 1 0", "0", "CAR", "Regression fixture",
             f"{nelect} {len(points)} 2"]
    for point in points:
        lines.extend(["", " ".join(map(str, point)) + " 0.25"])
        lines.extend(["1 -1 2", "2 2 0"] if ispin == 1 else
                     ["1 -1 -0.5 1 0.75", "2 2 0.8 0 0.25"])
    path = path_dir / "EIGENVAL"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
