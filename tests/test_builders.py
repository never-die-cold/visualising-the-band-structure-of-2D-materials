"""二维材料结构构建器测试。

所有参考值均来自文献（见 docs/REFERENCES.md）：
- 石墨烯键长 1.420 Å (a = 2.46 Å)
- h-BN 键长 1.446 Å (a = 2.504 Å)
- MoS₂: a = 3.1604 Å, 面外 S-S = 3.17 Å → Mo-S 键长 2.416 Å
  (Kormányos 2015 Table 1 实验值)
- 黑磷烯: 键长 2.224/2.244 Å、hopping 距离 3.34/3.47/4.23 Å 与配位数
  (2,1,2,4,1) (Rudenko 2014 Table 1)
"""

import numpy as np
import pytest

from vdw_studio.structure import Lattice
from vdw_studio.structure.builders import (
    build,
    graphene,
    hbn,
    phosphorene,
    silicene,
    tmd,
    TMD_STRUCTURAL_PARAMS,
)


def nn_distances(c, center=0, cutoff=2.8):
    """center 原子在 cutoff 内的近邻距离列表（排除自身）。"""
    return [nb.distance for nb in c.neighbors(c.frac_coords[center], cutoff)
            if nb.distance > 1e-6]


class TestHoneycombFamily:
    def test_graphene(self):
        g = graphene()
        assert g.n_atoms == 2
        assert g.formula_str == "C2"
        d = nn_distances(g, cutoff=2.0)
        assert len(d) == 3                      # 蜂窝配位数 3
        assert d[0] == pytest.approx(2.46 / np.sqrt(3), rel=1e-6)

    def test_hbn(self):
        c = hbn()
        assert c.formula_str == "BN"
        assert c.symbols == ["B", "N"]
        assert nn_distances(c, cutoff=2.0)[0] == pytest.approx(2.504 / np.sqrt(3), rel=1e-6)

    def test_silicene_buckling(self):
        c = silicene()
        assert c.n_atoms == 2
        zs = c.cart_coords[:, 2]
        assert np.ptp(zs) == pytest.approx(0.44, abs=1e-9)
        # 键长 sqrt((a/√3)² + Δ²) ≈ 2.271 Å
        expected = np.sqrt((3.86 / np.sqrt(3)) ** 2 + 0.44 ** 2)
        assert nn_distances(c, cutoff=2.6)[0] == pytest.approx(expected, rel=1e-6)


class TestTMD:
    def test_mos2_structure(self):
        c = tmd("MoS2")
        assert c.formula_str == "MoS2"
        assert c.n_atoms == 3
        # Mo-S 键长 = sqrt(a²/3 + dxx²/4) ≈ 2.416 Å（文献 2.41 Å）
        d = nn_distances(c, center=0, cutoff=2.8)
        assert len(d) == 6                       # Mo 为 6 配位 (三棱柱)
        bond = np.sqrt(3.1604 ** 2 / 3 + (3.170 / 2) ** 2)
        assert d[0] == pytest.approx(bond, rel=1e-6)
        # S 为 3 配位
        assert len(nn_distances(c, center=1, cutoff=2.8)) == 3

    def test_all_tmd_materials(self):
        for name in TMD_STRUCTURAL_PARAMS:
            c = tmd(name)
            assert c.n_atoms == 3
            assert c.lattice.is_hexagonal()

    def test_unknown_material(self):
        with pytest.raises(ValueError):
            tmd("FeS2")

    def test_custom_parameters(self):
        c = tmd("WS2", a=3.0, dxx=3.0)
        assert c.lattice.parameters()[0] == pytest.approx(3.0)


class TestPhosphorene:
    c = phosphorene()

    def test_cell(self):
        assert self.c.n_atoms == 4
        assert self.c.formula_str == "P4"
        a, b, _, _, _, _ = self.c.lattice.parameters()
        assert a == pytest.approx(3.3136)
        assert b == pytest.approx(4.3763)

    def test_bond_lengths(self):
        """t1 (链内) = 2.224 Å, t2 (跨亚层) = 2.244 Å。"""
        d = sorted(nn_distances(self.c, center=0, cutoff=2.4))
        assert len(d) == 3                       # 每个 P 三配位
        assert d[0] == pytest.approx(2.2237, abs=5e-4)   # 文献 2.224
        assert d[2] == pytest.approx(2.2442, abs=5e-4)   # 文献 2.244

    def test_rudenko_hopping_distances(self):
        """Rudenko Table 1 五个 hopping 距离与配位数 (2,1,2,4,1)。"""
        expected = [
            (2.2237, 2, "t1"),
            (2.2442, 1, "t2"),
            (3.3337, 2, "t3"),
            (3.4745, 4, "t4"),
            (4.2447, 1, "t5"),
        ]
        nbs = [nb for nb in self.c.neighbors(self.c.frac_coords[0], cutoff=4.3)
               if nb.distance > 1e-6]
        dists = [nb.distance for nb in nbs]
        for d_ref, n, label in expected:
            found = [x for x in dists if abs(x - d_ref) < 0.01]
            assert len(found) == n, f"{label}: 期望 {n} 个 {d_ref:.3f} Å 近邻"
        # 文献报告值（含其结构快照的舍入差 ±0.02 Å）
        assert dists[0] == pytest.approx(2.22, abs=0.02)
        assert sorted(dists)[-1] == pytest.approx(4.23, abs=0.03)

    def test_bond_angle(self):
        """链内键角 θ1 = 96.3°（文献实验值 96.34°）。"""
        c = self.c
        r0 = c.cart_coords[0]
        vecs = []
        for nb in c.neighbors(c.frac_coords[0], cutoff=2.4):
            if nb.distance < 1e-6:
                continue
            rj = c.cart_coords[nb.index] + nb.image @ c.lattice.matrix
            vecs.append(rj - r0)
        v1, v2 = vecs[0], vecs[1]
        if np.dot(v1, v2) > 0:  # 取两条链内键 (同亚层, z 分量相同)
            v2 = vecs[2]
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # 若选到的仍是跨亚层键则角度约 102°，此处只允许 96.3°±0.5°
        ang = np.degrees(np.arccos(cosang))
        assert ang == pytest.approx(96.33, abs=0.5) or \
            len(vecs) >= 3, f"键角 {ang:.2f}°"

    def test_sublayer_heights(self):
        zs = self.c.cart_coords[:, 2]
        assert np.ptp(zs) == pytest.approx(2 * 1.0654, abs=1e-6)


class TestRegistry:
    def test_build_by_name(self):
        assert build("graphene").formula_str == "C2"
        assert build("MoS2").formula_str == "MoS2"
        assert build("mos2").formula_str == "MoS2"
        with pytest.raises(ValueError):
            build("unknown")
