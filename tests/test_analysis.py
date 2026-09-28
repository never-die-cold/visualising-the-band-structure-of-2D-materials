"""性质分析模块测试（带隙/有效质量/费米速度/主质量）。

文献基准：
- 石墨烯：零带隙、v_F = √3|t|a/(2ħ) ≈ 8.74×10⁵ m/s
- hBN 二带模型：K 点带隙 = Δ（大质量 Dirac 解析性质），
  m* = 2ħ²Δ/(3t²a²) ≈ 0.39 m₀
- 黑磷烯：Γ 直接带隙 1.52 eV、主质量 (0.167, 0.849) m₀、各向异性 > 3×
"""

import numpy as np
import pytest

from vdw_studio.analysis import (
    analyze_gap,
    effective_mass,
    fermi_velocity,
    principal_masses,
)
from vdw_studio.engine.kpath import KPath
from vdw_studio.engine.models import HoneycombModel, PhosphoreneRudenko


class TestGraphene:
    g = HoneycombModel(a=2.46, t=-2.7)

    def test_zero_gap_at_K(self):
        r = analyze_gap(self.g, n_per_segment=30)
        assert r.gap is None and r.direct is None
        assert r.vbm_label == "K" and r.cbm_label == "K"

    def test_fermi_velocity(self):
        v = fermi_velocity(self.g, (1 / 3, 1 / 3), band=1,
                           direction=(1 / 3, -1 / 3))
        assert v == pytest.approx(8.74e5, rel=0.02)


class TestHBN:
    h = HoneycombModel(a=2.504, t=-2.7, delta_onsite=3.5)

    def test_direct_gap_at_K(self):
        r = analyze_gap(self.h, n_per_segment=30)
        assert r.gap == pytest.approx(3.5, abs=1e-8)
        assert r.direct is True
        assert r.vbm_label == "K" and r.cbm_label == "K"

    def test_effective_mass_analytic(self):
        """大质量 Dirac 解析质量 m* = 2ħ²Δ/(3t²a²)。"""
        m = effective_mass(self.h, (1 / 3, 1 / 3), band=1,
                           direction=(1 / 3, -1 / 3))
        expected = 2 * 7.619964 * 3.5 / (3 * 2.7 ** 2 * 2.504 ** 2)
        assert m == pytest.approx(expected, rel=1e-2)


class TestPhosphorene:
    p = PhosphoreneRudenko()

    def test_direct_gap_at_gamma(self):
        r = analyze_gap(self.p, n_per_segment=30)
        assert r.gap == pytest.approx(1.52, abs=1e-6)
        assert r.direct is True
        assert r.vbm_label == "Γ" and r.cbm_label == "Γ"

    def test_principal_masses(self):
        """主质量：轻 (armchair) ≈ 0.17 m₀、重 (zigzag) ≈ 0.85 m₀。"""
        m1, m2, _ = principal_masses(self.p, (0.0, 0.0), band=2, dk=1e-4)
        light, heavy = min(m1, m2), max(m1, m2)
        assert light == pytest.approx(0.167, rel=0.08)
        assert heavy == pytest.approx(0.849, rel=0.08)
        assert heavy / light > 3.0

    def test_effective_mass_consistency(self):
        """沿主轴方向的质量与主质量一致。"""
        m_x = effective_mass(self.p, (0, 0), band=2, direction=(1, 0))
        m_y = effective_mass(self.p, (0, 0), band=2, direction=(0, 1))
        m1, m2, _ = principal_masses(self.p, (0, 0), band=2, dk=1e-4)
        assert sorted([m_x, m_y]) == pytest.approx(sorted([m1, m2]), rel=1e-4)
