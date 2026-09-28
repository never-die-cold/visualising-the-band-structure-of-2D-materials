"""k 路径模块测试。"""

import numpy as np
import pytest

from vdw_studio.engine.kpath import KPath, high_symmetry_points, lattice_type
from vdw_studio.structure.builders import phosphorene
from vdw_studio.structure.lattice import Lattice


class TestLatticeType:
    def test_detection(self):
        assert lattice_type(Lattice.hexagonal(3.16)) == "hexagonal"
        assert lattice_type(Lattice.square(3.0)) == "square"
        assert lattice_type(Lattice.rectangular(3.3136, 4.3763)) == "rectangular"


class TestHexagonalPoints:
    def test_k_point_norm(self):
        """|K| = 4π/(3a)（六方 BZ 角点）。"""
        a = 2.46
        pts = high_symmetry_points(Lattice.hexagonal(a))
        recip = Lattice.hexagonal(a).reciprocal_matrix
        K = np.array([*pts["K"], 0.0]) @ recip
        assert np.linalg.norm(K) == pytest.approx(4 * np.pi / (3 * a), rel=1e-10)

    def test_m_point_norm(self):
        """|M| = |b1|/2 = 2π/(√3 a)。"""
        a = 2.46
        recip = Lattice.hexagonal(a).reciprocal_matrix
        pts = high_symmetry_points(Lattice.hexagonal(a))
        M = np.array([*pts["M"], 0.0]) @ recip
        assert np.linalg.norm(M) == pytest.approx(2 * np.pi / (np.sqrt(3) * a),
                                                  rel=1e-10)

    def test_default_path(self):
        kp = KPath.for_lattice(Lattice.hexagonal(2.46))
        assert kp.path == ["G", "M", "K", "G"]

    def test_invalid_point_name(self):
        with pytest.raises(ValueError):
            KPath.for_lattice(Lattice.hexagonal(2.46), path=["G", "Z"])


class TestGenerate:
    def test_sample_count(self):
        kp = KPath.for_lattice(Lattice.hexagonal(2.46))
        kcart, x_axis, ticks = kp.generate(Lattice.hexagonal(2.46), n_per_segment=20)
        # 3 段 × 20 + 末点
        assert kcart.shape == (61, 3)
        assert len(x_axis) == 61
        assert len(ticks) == 4

    def test_endpoints_are_high_symmetry(self):
        lat = Lattice.hexagonal(2.46)
        kp = KPath.for_lattice(lat)
        kcart, _, _ = kp.generate(lat, n_per_segment=10)
        recip = lat.reciprocal_matrix
        assert np.allclose(kcart[0], np.zeros(3), atol=1e-12)
        assert np.allclose(kcart[-1], np.zeros(3), atol=1e-12)
        # 中间 tick = M, K, 回到 Γ
        assert np.linalg.norm(kcart[10]) > 0
        assert np.linalg.norm(kcart[20]) > 0

    def test_monotonic_axis(self):
        lat = Lattice.rectangular(3.3136, 4.3763)
        kp = KPath.for_lattice(lat)
        _, x_axis, _ = kp.generate(lat, n_per_segment=30)
        assert np.all(np.diff(x_axis) >= -1e-12)

    def test_phosphorene_path(self):
        """黑磷烯（正交格子）默认路径 Γ-X-S-Y-Γ。"""
        lat = phosphorene().lattice
        kp = KPath.for_lattice(lat)
        assert kp.path == ["G", "X", "S", "Y", "G"]

    def test_invalid_segment_count(self):
        kp = KPath.for_lattice(Lattice.hexagonal(2.46))
        with pytest.raises(ValueError):
            kp.generate(Lattice.hexagonal(2.46), n_per_segment=1)
