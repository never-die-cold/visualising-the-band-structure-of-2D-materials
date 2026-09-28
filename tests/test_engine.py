"""紧束缚引擎物理验证测试。

每个模型的验证基准均来自文献（docs/REFERENCES.md）：

石墨烯 (t = −2.7 eV, a = 2.46 Å):
  - K 点 Dirac 精确为零（f(K) = 0）
  - 电子-空穴对称 E ↔ −E（手征对称性）
  - 费米速度 v_F = √3|t|a/(2ħ) ≈ 0.874×10⁶ m/s

h-BN（交错势 Δ）:
  - K 点带隙严格等于 |Δ|（大质量 Dirac 模型）

硅烯（电场调控）:
  - K 点带隙 = |E·Δz|（交错势严格结果）

黑磷烯 (Rudenko & Katsnelson 2014, t₁..t₅):
  - Γ 点本征值 {−6.04, −1.18, 0.34, 6.88} eV（由 hopping 表解析可算）
  - Γ 点直接带隙 1.52 eV（论文模型值；GW 参考 1.60 eV）
  - 强各向异性有效质量
"""

import numpy as np
import pytest

from vdw_studio.engine.kpath import KPath
from vdw_studio.engine.models import (
    HBAR2_OVER_2M0,
    BuckledHoneycombModel,
    HoneycombModel,
    PhosphoreneRudenko,
)
from vdw_studio.engine.solver import solve_bands, solve_dos

# J→eV 与 Å→m 的组合换算：1 eV·Å 的斜率对应 1.51927e5 m/s
SLOPE_TO_VELOCITY = 1.602176634e-19 / (1.054571817e-34 * 1e10)


class TestGraphene:
    g = HoneycombModel(a=2.46, t=-2.7)

    def test_dirac_point_at_K(self):
        e = self.g.energies_at((1 / 3, 1 / 3))
        assert e[0] == pytest.approx(0.0, abs=1e-10)
        assert e[1] == pytest.approx(0.0, abs=1e-10)

    def test_electron_hole_symmetry(self):
        """手征对称性：能带关于 E=0 对称。"""
        kf = np.random.default_rng(42).uniform(-0.5, 0.5, size=(25, 2))
        e = self.g.bands(kf)
        assert np.allclose(e[:, 0], -e[:, 1], atol=1e-10)

    def test_fermi_velocity(self):
        """v_F = √3|t|a/(2ħ) ≈ 0.874×10⁶ m/s。"""
        dk = 1e-5
        K = np.array([1 / 3, 1 / 3])
        G = np.array([0.0, 0.0])
        d = (G - K) / np.linalg.norm(G - K)
        e = self.g.energies_at(K + dk * d)
        slope_frac = e[1] / dk
        recip = self.g.lattice.reciprocal_matrix
        slope_cart = slope_frac / np.linalg.norm(d @ recip[:2, :2])
        v = slope_cart * SLOPE_TO_VELOCITY
        assert v == pytest.approx(8.74e5, rel=0.02)

    def test_bands_along_path(self):
        bs = solve_bands(self.g, KPath.for_lattice(self.g.lattice), 40)
        assert bs.energies.shape == (121, 2)
        # Γ 点能量 = ±3|t| = ±8.1 eV（3 个最近邻相干叠加）
        assert bs.energies[0, 1] == pytest.approx(3 * 2.7, rel=1e-6)
        assert bs.energies[0, 0] == pytest.approx(-3 * 2.7, rel=1e-6)
        # Dirac 点在路径上：|E| 的最小值为 0（K 点处两带相交）
        assert np.abs(bs.energies).min() == pytest.approx(0.0, abs=1e-8)
        # 电子-空穴对称 → 谱关于 0 对称
        assert bs.energies.min() == pytest.approx(-bs.energies.max(), abs=1e-8)

    def test_hamiltonian_hermitian(self):
        rng = np.random.default_rng(0)
        for kf in rng.uniform(-1, 1, (10, 2)):
            H = self.g.hamiltonian(kf)
            assert np.allclose(H, H.conj().T, atol=1e-12)


class TestHBN:
    def test_gap_equals_staggered_potential(self):
        for delta in (2.0, 3.5, 5.0):
            m = HoneycombModel(a=2.504, t=-2.7, delta_onsite=delta)
            e = m.energies_at((1 / 3, 1 / 3))
            assert (e[1] - e[0]) == pytest.approx(abs(delta), rel=1e-10)

    def test_no_gap_when_delta_zero(self):
        """Δ=0 时退化为石墨烯：K 点两带简并（零带隙）。"""
        m = HoneycombModel(a=2.504, delta_onsite=0.0)
        e = m.energies_at((1 / 3, 1 / 3))
        assert e[0] == pytest.approx(e[1], abs=1e-10)


class TestSiliceneField:
    def test_field_opens_gap(self):
        """K 点带隙 = |E·Δz|（交错势严格结果）。"""
        for field in (0.0, 0.5, 1.0, 2.0):
            m = BuckledHoneycombModel(electric_field=field)
            e = m.energies_at((1 / 3, 1 / 3))
            assert (e[1] - e[0]) == pytest.approx(abs(field * 0.44), rel=1e-10)

    def test_gap_tunable_continuously(self):
        gaps = [BuckledHoneycombModel(electric_field=f).energies_at((1/3, 1/3))[1]
                - BuckledHoneycombModel(electric_field=f).energies_at((1/3, 1/3))[0]
                for f in np.linspace(0, 3, 7)]
        assert np.all(np.diff(gaps) > 0)   # 单调开启


class TestPhosphorene:
    p = PhosphoreneRudenko()

    def test_gamma_eigenvalues(self):
        """Γ 点本征值 = hopping 表的解析结果。"""
        e = self.p.energies_at((0.0, 0.0))
        expected = [-6.04, -1.18, 0.34, 6.88]
        assert np.allclose(e, expected, atol=1e-8)

    def test_direct_gap_at_gamma(self):
        """Γ 点直接带隙 1.52 eV (Rudenko 2014 模型值)。"""
        e = self.p.energies_at((0.0, 0.0))
        assert (e[2] - e[1]) == pytest.approx(1.52, abs=1e-8)

    def test_global_gap_at_gamma(self):
        """全局带隙位于 Γ（黑磷烯为直接带隙半导体）。"""
        mesh = np.array([[i / 20, j / 20] for i in range(20) for j in range(20)])
        e = self.p.bands(mesh)
        vbm = e[:, 1].max()
        cbm = e[:, 2].min()
        assert (cbm - vbm) == pytest.approx(1.52, rel=5e-3)
        # VBM/CBM 都在 Γ (索引 (0,0))
        i_vbm = np.argmax(e[:, 1])
        i_cbm = np.argmin(e[:, 2])
        assert mesh[i_vbm].tolist() == [0.0, 0.0]
        assert mesh[i_cbm].tolist() == [0.0, 0.0]

    def test_anisotropic_masses(self):
        """强各向异性有效质量（Γ 点中心差分）。

        文献参考：单层黑磷电子质量 轻方向 ≈ 0.15–0.2 m₀、
        重方向 ≈ 0.8–1.1 m₀（四带模型直接给出的量级）。
        m* = ħ²/(d²E/dk²)，ħ²/(2m₀) = 3.80998 eV·Å²。
        """
        dE = 1e-4
        e0 = self.p.energies_at((0.0, 0.0))
        recip = self.p.lattice.reciprocal_matrix
        masses = {}
        for direction, name in (((1.0, 0.0), "zigzag_x"), ((0.0, 1.0), "armchair_y")):
            d = np.array(direction)
            ep = self.p.energies_at(d * dE)
            em = self.p.energies_at(-d * dE)
            curv = ((ep + em - 2 * e0) / dE**2)[2]          # CB 二阶导 (分数坐标)
            scale = np.linalg.norm(d @ recip[:2, :2]) ** 2  # 转笛卡尔
            masses[name] = 2 * HBAR2_OVER_2M0 / (curv / scale)
        # 轻方向 (armchair) ≈ 0.15–0.25 m₀，重方向 (zigzag) ≈ 0.6–1.3 m₀
        assert 0.10 < masses["armchair_y"] < 0.30
        assert 0.60 < masses["zigzag_x"] < 1.30
        assert masses["zigzag_x"] / masses["armchair_y"] > 3.0

    def test_hermitian(self):
        rng = np.random.default_rng(1)
        for kf in rng.uniform(-1, 1, (10, 2)):
            H = self.p.hamiltonian(kf)
            assert np.allclose(H, H.conj().T, atol=1e-12)


class TestSolver:
    def test_dos_graphene_integrates_to_states(self):
        """DOS 积分 ≈ 每原胞态数 2（二带模型）。"""
        g = HoneycombModel(a=2.46, t=-2.7)
        dos = solve_dos(g, mesh=(40, 40), sigma=0.10, n_points=1200)
        integral = np.trapezoid(dos.dos, dos.energies)
        assert integral == pytest.approx(2.0, rel=0.02)

    def test_dos_graphene_vanishes_at_dirac(self):
        """石墨烯 Dirac 点 DOS→0（半金属特征）。"""
        g = HoneycombModel(a=2.46, t=-2.7)
        dos = solve_dos(g, mesh=(48, 48), sigma=0.05, n_points=1500)
        i0 = np.argmin(np.abs(dos.energies))
        assert dos.dos[i0] < 0.1 * dos.dos.max()

    def test_dos_phosphorene_gap(self):
        """磷烯 DOS 在带隙深处 (−0.5, +0.15) eV 应为零（Γ 带隙 1.52 eV）。"""
        p = PhosphoreneRudenko()
        dos = solve_dos(p, mesh=(36, 36), sigma=0.03, n_points=1200,
                        e_min=-0.8, e_max=0.8)
        mask = (dos.energies > -0.5) & (dos.energies < 0.15)
        assert dos.dos[mask].max() < 1e-3
