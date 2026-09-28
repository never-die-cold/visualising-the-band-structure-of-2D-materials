"""结构导出 (POSCAR / XYZ) 测试。"""

import numpy as np
import pytest

from vdw_studio.io import poscar_string, write_poscar, xyz_string
from vdw_studio.structure.builders import build, phosphorene, tmd


class TestPOSCAR:
    def test_mos2_poscar_format(self):
        c = tmd("MoS2")
        s = poscar_string(c)
        lines = s.strip().splitlines()
        assert lines[0] == "MoS2"
        assert lines[1] == "1.0"
        # 3 条晶格矢量
        lat = [[float(x) for x in ln.split()] for ln in lines[2:5]]
        assert np.allclose(lat[0], [3.1604, 0, 0], atol=1e-9)
        # 元素行 + 数量行
        assert lines[5].split() == ["Mo", "S"]
        assert lines[6].split() == ["1", "2"]
        assert lines[7].strip() == "Direct"
        assert len(lines) == 11

    def test_cartesian_mode(self):
        c = phosphorene()
        s = poscar_string(c, title="black phosphorene", direct=False)
        lines = s.strip().splitlines()
        assert lines[0] == "black phosphorene"
        cart = np.array([[float(x) for x in ln.split()] for ln in lines[8:12]])
        assert np.allclose(cart, c.cart_coords, atol=1e-9)

    def test_supercell_export(self):
        c = build("graphene").supercell((2, 2, 1))
        s = poscar_string(c)
        lines = s.strip().splitlines()
        assert lines[5].split() == ["C"]
        assert lines[6].split() == ["8"]
        a_lat = [float(x) for x in lines[2].split()]
        assert a_lat[0] == pytest.approx(4.92)

    def test_write_file_roundtrip(self, tmp_path):
        c = tmd("WSe2")
        p = tmp_path / "POSCAR"
        write_poscar(c, str(p))
        text = p.read_text(encoding="utf-8")
        lines = text.strip().splitlines()
        assert lines[5].split() == ["W", "Se"]
        # 文件可被重新解析（简单自解析校验）
        lat = np.array([[float(x) for x in ln.split()] for ln in lines[2:5]])
        assert lat.shape == (3, 3)
        assert np.allclose(lat, c.lattice.matrix, atol=1e-9)


class TestXYZ:
    def test_xyz_format(self):
        c = tmd("MoS2")
        s = xyz_string(c)
        lines = s.strip().splitlines()
        assert lines[0] == "3"
        assert lines[1].startswith("MoS2")
        assert lines[2].split()[0] == "Mo"
        cart = np.array([[float(x) for x in ln.split()[1:]] for ln in lines[2:5]])
        assert np.allclose(cart, c.cart_coords, atol=1e-8)
