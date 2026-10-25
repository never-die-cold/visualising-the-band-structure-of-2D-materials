"""通用多层堆叠构建器的结构与几何验证测试。

验证锚点全部可解析：
- 2H n=2 与 ``builders.tmd_bilayer`` 逐坐标等价（回归锚点）；
- 2H-MoS₂ 跨层最近邻 S–S ≈ 3.49 Å vdW 接触、Mo–Mo = interlayer；
- Bernal AB：上层 A 位在下层 B 位正上方（零面内偏移的二聚位）；
- 3R：层同向、位移 (0,0)→(1/3,2/3)→(2/3,1/3) 三层周期；
- 转角 (m,n)：cos θ = (m²+4mn+n²)/(2N)、每层 N 个原胞、层内键长不变。
"""

import math
from itertools import product

import numpy as np
import pytest

from vdw_studio.structure.builders import graphene, tmd, tmd_bilayer
from vdw_studio.structure.multilayer import (
    LayerStep,
    stack_2h,
    stack_3r,
    stack_bernald,
    stack_sequence,
    stack_twisted,
)

TMD_BILAYER_INTERLAYER = 6.15   # builders.TMD_BILAYER_INTERLAYER


def _inplane_frac(crystal, indices):
    """面内分数坐标（折回 [0,1)）。"""
    f = crystal.frac_coords[indices][:, :2]
    return np.mod(f, 1.0)


def _match_inplane(fracs_a, fracs_b, lattice, tol=1e-8):
    """两组面内分数坐标是否为同一集合（允许整体平移与排列差异）。"""
    used = set()
    for a in fracs_a:
        d = np.mod(fracs_b - a + 0.5, 1.0) - 0.5
        dist = np.linalg.norm(d @ lattice.matrix[:2, :2], axis=1)
        order = np.argsort(dist)
        placed = False
        for j in order:
            if int(j) not in used and dist[j] < tol:
                used.add(int(j))
                placed = True
                break
        if not placed:
            return False
    return True


class TestSequence:
    def test_single_layer_preserves_inplane_geometry(self):
        mono = tmd("MoS2")
        s = stack_sequence(mono, [LayerStep()], 6.15)
        assert s.crystal.n_atoms == 3
        assert s.layer_slices == ((0, 3),)
        # 面内相对坐标不变（整体平移 (1/3,2/3) 允许）
        rel_mono = np.mod(mono.frac_coords[:, :2] - mono.frac_coords[0, :2],
                          1.0)
        rel_stack = np.mod(s.crystal.frac_coords[:, :2]
                           - s.crystal.frac_coords[0, :2], 1.0)
        d = np.mod(rel_stack - rel_mono + 0.5, 1.0) - 0.5
        assert np.linalg.norm(d @ s.crystal.lattice.matrix[:2, :2]) < 1e-8

    def test_explicit_shifts_move_layers(self):
        mono = graphene()
        s = stack_sequence(mono, [LayerStep(), LayerStep(shift=(1 / 3, 2 / 3))],
                           3.35)
        f = _inplane_frac(s.crystal, slice(None))
        # 层 2 = 层 1 + (1/3, 2/3)
        d = np.mod(f[2:] - f[:2] - np.array([1 / 3, 2 / 3]) + 0.5, 1.0) - 0.5
        assert np.abs(d).max() < 1e-9

    def test_rotation_step_flips_layer(self):
        mono = tmd("MoS2")
        s = stack_sequence(mono, [LayerStep(), LayerStep(rot180=True)], 6.15)
        # rot180 后 Mo 的面内位置 (0,0) 不动，两个 S 的空位互换：
        # (1/3,2/3) → (2/3,1/3)
        f = _inplane_frac(s.crystal, slice(3, 6))
        mo, s_top, s_bot = f[0], f[1], f[2]
        assert np.allclose(np.mod(mo - s_top + 0.5, 1.0) - 0.5,
                           np.mod(np.array([-2 / 3, -1 / 3]) + 0.5, 1.0) - 0.5,
                           atol=1e-9)
        assert np.allclose(np.mod(s_top - s_bot, 1.0), 0.0, atol=1e-9)

    def test_invalid_inputs(self):
        mono = tmd("MoS2")
        with pytest.raises(ValueError):
            stack_sequence(mono, [], 6.15)
        with pytest.raises(ValueError):
            stack_sequence(mono, [LayerStep()], 0.0)
        with pytest.raises(ValueError):
            stack_sequence(mono, [LayerStep()], 6.15, vacuum=10.0)
        with pytest.raises(ValueError):
            stack_2h(mono, 0, 6.15)
        with pytest.raises(ValueError):
            stack_2h("not a crystal", 2, 6.15)


class Test2H:
    def test_bilayer_matches_tmd_bilayer_builder(self):
        mono = tmd("MoS2")
        s = stack_2h(mono, 2, TMD_BILAYER_INTERLAYER)
        ref = tmd_bilayer("MoS2")
        assert s.crystal.n_atoms == ref.n_atoms == 6
        for i in range(6):
            fa = _inplane_frac(s.crystal, [i])[0]
            fb = _inplane_frac(ref, [i])[0]
            d = np.mod(fa - fb + 0.5, 1.0) - 0.5
            assert np.linalg.norm(d) < 1e-9
            # 相对胞中心的 z 偏移一致（真空约定不同，比相对值）
            za = (s.crystal.frac_coords[i, 2] - 0.5) * s.lattice.volume \
                / s.lattice.area_2d
            zb = (ref.frac_coords[i, 2] - 0.5) * ref.lattice.volume \
                / ref.lattice.area_2d
            assert za == pytest.approx(zb, abs=1e-9)

    def test_cross_layer_distances(self):
        s = stack_2h(tmd("MoS2"), 2, TMD_BILAYER_INTERLAYER)
        cart = s.crystal.cart_coords
        # Mo–Mo 的**垂直分量** = interlayer（与 builders.tmd_bilayer 同约定；
        # 3D 距离含 a/√3 面内错位，为 sqrt(6.15² + 1.825²) ≈ 6.41 Å）
        d_momo = cart[3] - cart[0]
        assert abs(d_momo[2]) == pytest.approx(TMD_BILAYER_INTERLAYER, rel=1e-9)
        assert np.linalg.norm(d_momo[:2]) == pytest.approx(
            3.1604 / math.sqrt(3), rel=1e-9)
        # 跨层最近 S–S ≈ sqrt((c/2−dxx)² + (a/√3)²) ≈ 3.49 Å vdW 接触
        expected = math.sqrt((TMD_BILAYER_INTERLAYER - 3.170) ** 2
                             + (3.1604 / math.sqrt(3)) ** 2)
        d_ss = min(np.linalg.norm(cart[4 + k] - cart[1 + j])
                   for k in range(2) for j in range(2))
        assert d_ss == pytest.approx(expected, rel=2e-3)
        # 层内键长与单层一致（Mo–S ≈ 2.41 Å）
        mono_cart = tmd("MoS2").cart_coords
        bond_mono = np.linalg.norm(mono_cart[1] - mono_cart[0])
        bond_stack = np.linalg.norm(cart[1] - cart[0])
        assert bond_stack == pytest.approx(bond_mono, rel=1e-9)

    def test_two_layer_periodicity(self):
        s = stack_2h(tmd("MoS2"), 4, TMD_BILAYER_INTERLAYER)
        # 层 3 与层 1 同位形（AbAB 序列的周期性）
        assert _match_inplane(_inplane_frac(s.crystal, slice(0, 3)),
                              _inplane_frac(s.crystal, slice(6, 9)),
                              s.crystal.lattice)
        # 层 2 与层 4 同位形
        assert _match_inplane(_inplane_frac(s.crystal, slice(3, 6)),
                              _inplane_frac(s.crystal, slice(9, 12)),
                              s.crystal.lattice)


class TestBernal:
    def test_dimer_site_alignment(self):
        """上层 A 位在下层 B 位正上方（二聚位，γ₁ 耦合位形）。"""
        s = stack_bernald(graphene(), 2, 3.35)
        f = _inplane_frac(s.crystal, slice(None))
        cart = s.crystal.cart_coords
        # 层 2 的 A (index 2) 与层 1 的 B (index 1) 面内重合
        d = np.mod(f[2] - f[1] + 0.5, 1.0) - 0.5
        assert np.linalg.norm(d) < 1e-9
        # 且面外距离 = interlayer
        assert np.linalg.norm(cart[2] - cart[1]) == pytest.approx(3.35,
                                                                  rel=1e-9)
        # 另一个子格（B2）在空位上方：与层 1 A 的面内距离 = a/√3
        # （分数差必须经晶格度量转笛卡尔，不能直接乘 |a1|）
        d_frac = np.mod(f[3] - f[0] + 0.5, 1.0) - 0.5
        d_cart = np.linalg.norm(d_frac @ s.crystal.lattice.matrix[:2, :2])
        assert d_cart == pytest.approx(2.46 / math.sqrt(3), rel=1e-6)

    def test_abab_periodicity(self):
        s = stack_bernald(graphene(), 3, 3.35)
        assert _match_inplane(_inplane_frac(s.crystal, slice(0, 2)),
                              _inplane_frac(s.crystal, slice(4, 6)),
                              s.crystal.lattice)


class Test3R:
    def test_shift_cycle_same_orientation(self):
        s = stack_3r(tmd("MoS2"), 4, TMD_BILAYER_INTERLAYER)
        # 每层同向：层内 Mo–S 面内位移矢量在四层中逐分量相同（无翻转）
        dvecs = []
        for li in range(4):
            f = _inplane_frac(s.crystal, s.layer_atoms(li))
            dvecs.append(np.mod(f[1] - f[0] + 0.5, 1.0) - 0.5)
        for d in dvecs[1:]:
            assert np.allclose(d, dvecs[0], atol=1e-9)
        # 面内 Mo–S 偏移的模 = a/√3（经晶格度量）
        d_cart = np.linalg.norm(dvecs[0] @ s.crystal.lattice.matrix[:2, :2])
        assert d_cart == pytest.approx(3.1604 / math.sqrt(3), rel=1e-6)
        # 位移序列 (0,0) → (1/3,2/3) → (2/3,1/3) → (0,0)：Mo 的面内位置
        mo_pos = [np.mod(_inplane_frac(s.crystal, s.layer_atoms(li))[0], 1.0)
                  for li in range(4)]
        expected = np.array([[0.0, 0.0], [1 / 3, 2 / 3],
                             [2 / 3, 1 / 3], [0.0, 0.0]])
        for got, want in zip(mo_pos, expected):
            assert np.allclose(np.mod(got - want + 0.5, 1.0) - 0.5, 0.0,
                               atol=1e-9)


class TestTwisted:
    def test_21_coincidence_cell(self):
        st = stack_twisted(graphene(), 2, 1, 3.35)
        assert st.crystal.n_atoms == 2 * 2 * 7
        theta = st.meta["theta_deg"]
        assert theta == pytest.approx(math.degrees(math.acos(13 / 14)),
                                      rel=1e-9)
        a = 2.46
        assert np.linalg.norm(st.lattice.a1[:2]) == pytest.approx(
            a * math.sqrt(7), rel=1e-9)
        # 层内最近邻键长保持 a/√3（层未被破坏）
        for lo, hi in st.layer_slices:
            xy = st.crystal.frac_coords[lo:hi]
            dmin = min(np.linalg.norm((xy[j] + np.array([p, q, 0]) - xy[i])
                                      @ st.lattice.matrix)
                       for i in range(hi - lo) for j in range(hi - lo)
                       for p, q in product((-1, 0, 1), (-1, 0, 1))
                       if i != j or p or q)
            assert dmin == pytest.approx(a / math.sqrt(3), rel=1e-6)

    def test_10_is_60_degree(self):
        st = stack_twisted(tmd("MoS2"), 1, 0, TMD_BILAYER_INTERLAYER)
        assert st.crystal.n_atoms == 2 * 3 * 1
        assert st.meta["theta_deg"] == pytest.approx(60.0, rel=1e-9)

    def test_32_cells(self):
        st = stack_twisted(tmd("MoS2"), 3, 2, TMD_BILAYER_INTERLAYER)
        assert st.crystal.n_atoms == 2 * 3 * 19
        assert st.meta["theta_deg"] == pytest.approx(
            math.degrees(math.acos(37 / 38)), rel=1e-9)
        assert st.meta["cells_per_layer"] == 19

    def test_invalid_pairs(self):
        with pytest.raises(ValueError):
            stack_twisted(graphene(), 2, 2, 3.35)   # gcd = 2
        with pytest.raises(ValueError):
            stack_twisted(graphene(), 0, 0, 3.35)
