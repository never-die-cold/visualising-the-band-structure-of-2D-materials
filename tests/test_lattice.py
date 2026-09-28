"""Lattice 晶格模块测试。"""

import numpy as np
import pytest

from vdw_studio.structure import Lattice


class TestConstructors:
    def test_hexagonal_parameters(self):
        a, vac = 3.16, 15.0
        lat = Lattice.hexagonal(a, vacuum=vac)
        la, lb, lc, al, be, ga = lat.parameters()
        assert la == pytest.approx(a)
        assert lb == pytest.approx(a)
        assert lc == pytest.approx(vac)
        assert ga == pytest.approx(120.0, abs=1e-6)
        assert al == pytest.approx(90.0, abs=1e-6)

    def test_hexagonal_area(self):
        a = 3.16
        lat = Lattice.hexagonal(a)
        assert lat.area_2d == pytest.approx(a * a * np.sqrt(3) / 2)

    def test_rectangular(self):
        lat = Lattice.rectangular(3.32, 10.47)
        assert lat.parameters()[:2] == pytest.approx((3.32, 10.47))

    def test_invalid_matrix(self):
        with pytest.raises(ValueError):
            Lattice(np.zeros((3, 3)))
        with pytest.raises(ValueError):
            Lattice(np.eye(2))

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            Lattice.from_parameters(-1, 2, 3)
        # 夹角和 >= 360° 无意义
        with pytest.raises(ValueError):
            Lattice.from_parameters(1, 1, 1, 170, 170, 170)
        # 全部夹角过于尖锐，z 分量无解 (cz² < 0)
        with pytest.raises(ValueError):
            Lattice.from_parameters(1, 1, 1, 30, 30, 170)


class TestReciprocal:
    def test_hexagonal_reciprocal_norms(self):
        """六方晶格倒格矢长度 = 4π/(√3 a)。"""
        a = 3.16
        lat = Lattice.hexagonal(a)
        recip = lat.reciprocal_matrix
        expected = 4 * np.pi / (np.sqrt(3) * a)
        for i in range(2):
            assert np.linalg.norm(recip[i]) == pytest.approx(expected, rel=1e-10)

    def test_reciprocal_orthogonality_honeycomb_plane(self):
        """实空间 γ=120° → 倒空间夹角 60°，故 b1·b2 = |b|²/2 > 0。"""
        lat = Lattice.hexagonal(2.46)
        b = lat.reciprocal_matrix
        n1, n2 = np.linalg.norm(b[0]), np.linalg.norm(b[1])
        assert n1 == pytest.approx(n2)
        assert np.dot(b[0], b[1]) == pytest.approx(n1 * n2 / 2)

    def test_cubic_reciprocal(self):
        lat = Lattice.square(5.0, vacuum=20.0)
        recip = lat.reciprocal_matrix
        assert np.linalg.norm(recip[0]) == pytest.approx(2 * np.pi / 5.0)
        assert np.linalg.norm(recip[1]) == pytest.approx(2 * np.pi / 5.0)
        assert np.dot(recip[0], recip[1]) == pytest.approx(0.0, abs=1e-10)

    def test_reciprocal_lattice_object_roundtrip(self):
        """r = f·A 与 k = g·B 相互独立；倒格子对象可复用几何方法。"""
        lat = Lattice.hexagonal(3.16)
        recip_lat = lat.reciprocal()
        assert isinstance(recip_lat, Lattice)
        assert recip_lat.parameters()[:3] == pytest.approx(
            (4 * np.pi / (np.sqrt(3) * 3.16),) * 2 + (2 * np.pi / 15.0,)
        )


class TestCoordinateTransforms:
    def test_frac_cart_roundtrip(self):
        lat = Lattice.hexagonal(3.16)
        f = np.array([[0.1, 0.3, 0.5], [0.7, 0.9, 0.2]])
        cart = lat.frac_to_cart(f)
        back = lat.cart_to_frac(cart)
        assert np.allclose(back, f, atol=1e-12)

    def test_bijection_with_integer_lattice_vectors(self):
        """整数分数坐标位移 = 平移一个晶格矢量。"""
        lat = Lattice.rectangular(3.32, 10.47)
        r = np.array([0.2, 0.4, 0.1])
        shifted = lat.frac_to_cart(r + np.array([1.0, 0.0, 0.0]))
        assert np.allclose(shifted - lat.frac_to_cart(r), lat.a1)


class TestStrain:
    def test_biaxial_scaling(self):
        lat = Lattice.hexagonal(3.16)
        strained = lat.scaled_xy(1.05, 1.05)
        assert strained.parameters()[0] == pytest.approx(3.16 * 1.05)
        assert strained.parameters()[2] == pytest.approx(15.0)  # 真空层不变

    def test_uniaxial(self):
        lat = Lattice.rectangular(3.32, 10.47)
        strained = lat.scaled_xy(1.0, 0.95)
        assert strained.parameters()[0] == pytest.approx(3.32)
        assert strained.parameters()[1] == pytest.approx(10.47 * 0.95)

    def test_original_untouched(self):
        lat = Lattice.hexagonal(3.16)
        _ = lat.scaled_xy(2.0, 2.0)
        assert lat.parameters()[0] == pytest.approx(3.16)


def test_repr_and_eq():
    a = Lattice.hexagonal(3.16)
    b = Lattice.hexagonal(3.16, vacuum=15.0)
    assert a == b
    assert a != Lattice.hexagonal(3.20)
    assert "3.16" in repr(a)
