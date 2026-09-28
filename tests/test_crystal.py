"""Crystal 晶体结构模块测试。"""

import numpy as np
import pytest

from vdw_studio.structure import Atom, Crystal, Lattice


def make_simple_cubic(a=2.0, n=1):
    lat = Lattice.square(a, vacuum=2.0 * a)
    c = Crystal(lat)
    for i in range(n):
        c.add_atom("C", (0.0, 0.0, 0.1 * i))
    return c


class TestAtom:
    def test_frac_validation(self):
        a = Atom("Mo", (0.0, 0.0, 0.5))
        assert a.frac.shape == (3,)
        with pytest.raises(ValueError):
            Atom("Xx", (0, 0, 0))

    def test_label(self):
        a = Atom("S", (1 / 3, 2 / 3, 0.6), label="S1")
        assert a.label == "S1"


class TestCrystalBasics:
    def test_formula_and_hill_notation(self):
        c = make_simple_cubic()
        c.add_atom("Mo", (0.5, 0.5, 0.5))
        c.add_atom("S", (1 / 3, 2 / 3, 0.3))
        c.add_atom("S", (2 / 3, 1 / 3, 0.7))
        assert c.n_atoms == 4
        assert c.formula_str == "CMoS2"
        assert c.formula == {"C": 1, "Mo": 1, "S": 2}

    def test_empty(self):
        c = make_simple_cubic()
        c2 = Crystal(c.lattice)
        assert c2.n_atoms == 0
        assert c2.formula_str == ""

    def test_invalid_symbol_rejected(self):
        c = make_simple_cubic()
        with pytest.raises(ValueError):
            c.add_atom("Zz", (0.1, 0.1, 0.1))

    def test_copy_and_wrap(self):
        c = make_simple_cubic()
        c.add_atom("C", (1.3, -0.2, 0.5))
        c2 = c.copy()
        c2.wrap()
        assert np.allclose(c2.frac_coords, np.mod(c.frac_coords, 1.0))
        assert c.frac_coords[1][0] == pytest.approx(1.3)  # 原结构未被修改


class TestNeighbors:
    def test_nearest_neighbors_graphene_cell(self):
        """石墨烯原胞内最近邻：每个原子应找到 1 个胞内近邻（距离 1.42 Å）。"""
        a = 2.46
        lat = Lattice.hexagonal(a)
        c = Crystal(lat)
        c.add_atom("C", (0.0, 0.0, 0.5))
        c.add_atom("C", (1 / 3, 2 / 3, 0.5))
        nbs = c.neighbors(c.frac_coords[0], cutoff=1.6)
        dists = [nb.distance for nb in nbs if nb.distance > 1e-6]
        assert len(dists) == 3  # 周期镜像下每个原子有 3 个最近邻
        for d in dists:
            assert d == pytest.approx(a / np.sqrt(3), rel=1e-6)

    def test_neighbors_include_images(self):
        c = make_simple_cubic(a=2.0)
        nbs = c.neighbors((0.0, 0.0, 0.1), cutoff=2.5)
        # 自身 + 各方向周期镜像
        assert any(nb.image.tolist() == [1, 0, 0] for nb in nbs)
        assert any(nb.image.tolist() == [0, -1, 0] for nb in nbs)

    def test_sorted_by_distance(self):
        c = make_simple_cubic(a=2.0)
        c.add_atom("C", (0.5, 0.5, 0.1))
        nbs = c.neighbors((0.0, 0.0, 0.1), cutoff=2.5)
        dists = [nb.distance for nb in nbs]
        assert dists == sorted(dists)


class TestBonds:
    def test_graphene_bond_count(self):
        """2×2 石墨烯超胞：8 个原子、12 根键（配位数 3）。"""
        a = 2.46
        c = Crystal(Lattice.hexagonal(a))
        c.add_atom("C", (0.0, 0.0, 0.5))
        c.add_atom("C", (1 / 3, 2 / 3, 0.5))
        sc = c.supercell((2, 2, 1))
        bonds = sc.bonds()
        assert len(bonds) == 12
        coord = np.zeros(sc.n_atoms)
        for i, j, _, _ in bonds:
            coord[i] += 1
            coord[j] += 1
        assert np.all(coord == 3)

    def test_no_bond_across_vacuum(self):
        """两层石墨烯隔 5 Å 真空不应成键。"""
        c = Crystal(Lattice.hexagonal(2.46, vacuum=8.0))
        c.add_atom("C", (0.0, 0.0, 0.3))
        c.add_atom("C", (1 / 3, 2 / 3, 0.3))
        c.add_atom("C", (0.0, 0.0, 0.9))
        c.add_atom("C", (1 / 3, 2 / 3, 0.9))
        zs = c.cart_coords[:, 2]
        # z 方向间隔 = 8 × 0.6 = 4.8 Å，远超 C-C 成键判据
        assert np.ptp(zs) == pytest.approx(4.8)
        bonds = c.bonds()
        for i, j, img, dist in bonds:
            assert abs(c.cart_coords[i][2] - (c.cart_coords[j][2] + img[2] * 8.0)) < 2.0


class TestSupercell:
    def test_supercell_counts_and_lattice(self):
        c = make_simple_cubic(a=2.0, n=2)
        sc = c.supercell((3, 2, 1))
        assert sc.n_atoms == 12
        assert sc.lattice.parameters()[0] == pytest.approx(6.0)
        assert sc.lattice.parameters()[1] == pytest.approx(4.0)
        assert sc.formula_str == "C12"

    def test_supercell_density_invariant(self):
        """超胞后原子数密度不变。"""
        a, vac = 3.16, 15.0
        c = Crystal(Lattice.hexagonal(a, vacuum=vac))
        c.add_atom("Mo", (0, 0, 0.5))
        sc = c.supercell((2, 2, 1))
        rho1 = c.n_atoms / c.lattice.volume
        rho2 = sc.n_atoms / sc.lattice.volume
        assert rho1 == pytest.approx(rho2)

    def test_invalid_rep(self):
        c = make_simple_cubic()
        with pytest.raises(ValueError):
            c.supercell((0, 1, 1))
