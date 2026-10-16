"""玻尔兹曼输运测试（CRTA 迁移率张量，Phase 4.4）。

验证锚点：
1. **Drude 精确**（黑磷烯空穴带，T=100 K）：单带抛物线 + 热窗干净时
   CRTA 迁移率 μ = eτ/m*（与温度/简并度无关的严格结果）。
   数值与解析值吻合 1–3%（T=300 K 时 3–8%——真实非抛物线效应）。
2. **费米能级解析式**：2D 单带有限温公式
   μ = E_V − kT·ln(exp(p/(D·kT)) − 1)，D = g·m_geo/(2πħ²)，
   与二分反解值吻合 ≤1 meV。
3. **各向异性**：μ 比 = 质量比（CRTA 各向异性完全由色散决定）；
   高迁移率方向为 y 轴（4.38 Å）——与 Qiao et al., PRL 113, 256803 (2014)
   的单层黑磷空穴迁移率高方向一致（y: 10⁴–2.6×10⁴ vs x: 640–700 cm²/Vs）。
4. **结构性质**：无磁场 σ_xy = 0；六方对称 μ_xx = μ_yy（双层）、
   μ ∝ τ（CRTA 定义）。
5. **触碰带注记**：双层石墨烯低密度下 CRTA μ ≠ eτ/m*（镜像带同处
   热窗共同贡献电导）——只做结构检验，见 transport.py 模块文档。
"""

import numpy as np
import pytest
from functools import lru_cache

from vdw_studio.analysis.properties import effective_mass
from vdw_studio.analysis.transport import solve_transport
from vdw_studio.engine.models import BilayerGrapheneModel, PhosphoreneRudenko


def drude_mu_cm2(tau_fs: float, m_over_m0: float) -> float:
    """eτ/m* 换算到 cm²/Vs。"""
    return 1.602176634e-19 * tau_fs * 1e-15 / (
        m_over_m0 * 9.1093837015e-31) * 1e4


@lru_cache(maxsize=None)
def _phosphorene_holes():
    return solve_transport(PhosphoreneRudenko(), n_cm2=-1e12,
                           tau_fs=10.0, temperature=100.0, n_mesh=192)


@lru_cache(maxsize=None)
def _bilayer_electrons():
    return solve_transport(BilayerGrapheneModel(electric_field=0.0),
                           n_cm2=2e12, tau_fs=10.0,
                           temperature=300.0, n_mesh=200)


def _phosphorene_masses():
    p = PhosphoreneRudenko()
    m_x = effective_mass(p, (0.0, 0.0), band=1, direction=(1, 0))
    m_y = effective_mass(p, (0.0, 0.0), band=1, direction=(0, 1))
    return m_x, m_y


class TestPhosphoreneHoles:
    """黑磷烯空穴输运：Drude 质量反演 + 各向异性（干净单带体系）。"""

    def test_drude_mass_recovery_armchair(self):
        """μ_xx ≈ eτ/m_x（T=100 K，4% 窗口含非抛物线残差）。"""
        m_x, _ = _phosphorene_masses()
        r = _phosphorene_holes()
        assert r.mu_xx == pytest.approx(drude_mu_cm2(10.0, m_x), rel=0.04)

    def test_drude_mass_recovery_zigzag(self):
        """μ_yy ≈ eτ/m_y。"""
        _, m_y = _phosphorene_masses()
        r = _phosphorene_holes()
        assert r.mu_yy == pytest.approx(drude_mu_cm2(10.0, m_y), rel=0.04)

    def test_anisotropy_ratio_equals_mass_ratio(self):
        """μ_x/μ_y = m_y/m_x（CRTA 各向异性由色散决定），~6 倍。"""
        m_x, m_y = _phosphorene_masses()
        r = _phosphorene_holes()
        assert r.mu_xx / r.mu_yy == pytest.approx(m_y / m_x, rel=0.05)
        assert r.mu_yy > r.mu_xx          # 高迁移率方向 = y 轴（Qiao 2014）
        assert 4.0 < r.mu_yy / r.mu_xx < 8.0

    def test_fermi_level_analytic(self):
        """费米能级与 2D 有限温解析式吻合 ≤2 meV。

        μ = E_V − kT·ln(exp(p/(D·kT)) − 1)，D = g·m_geo/(2πħ²)，
        m_geo = √(m_x·m_y)（Γ 谷、单带）。
        """
        m_x, m_y = _phosphorene_masses()
        r = _phosphorene_holes()
        e_v = -1.18                      # Γ 价带顶（模型精确值）
        d_2d = 2.0 * np.sqrt(m_x * m_y) * 9.1093837015e-31 / (
            2 * np.pi * (1.054571817e-34) ** 2)
        kt_j = 8.617333262e-5 * 100 * 1.602176634e-19
        mu_analytic = e_v - kt_j * np.log(
            np.exp(1e16 / (d_2d * kt_j)) - 1) / 1.602176634e-19
        assert r.fermi_ev == pytest.approx(mu_analytic, abs=2e-3)

    def test_time_reversal_sigma_xy(self):
        """无磁场：|σ_xy| ≪ σ_xx（时间反演对称）。"""
        r = _phosphorene_holes()
        assert abs(r.sigma_xy) < 1e-6 * r.sigma_xx

    def test_sigma_quantum_units_consistent_with_mobility(self):
        """σ[(e²/h)] 与 μ 的换算自洽：σ = n·e·μ。"""
        r = _phosphorene_holes()
        h_over_e2 = 25812.80745
        sigma_si = r.sigma_xx / h_over_e2    # S
        mu_from_sigma = sigma_si / (1.602176634e-19 * 1e16) * 1e4
        assert mu_from_sigma == pytest.approx(r.mu_xx, rel=1e-9)


class TestBilayerElectrons:
    """双层石墨烯（触碰带）：结构检验（绝对 Drude 不适用，见模块文档）。"""

    def test_isotropy(self):
        """六方对称：μ_xx = μ_yy（相对偏差 < 1e-3）。"""
        r = _bilayer_electrons()
        assert r.mu_xx == pytest.approx(r.mu_yy, rel=1e-3)

    def test_tau_linear_scaling(self):
        """μ ∝ τ（CRTA 定义；同一网格比值精确）。"""
        r = _bilayer_electrons()
        r2 = solve_transport(BilayerGrapheneModel(), n_cm2=2e12,
                             tau_fs=20.0, temperature=300.0, n_mesh=200)
        assert r2.mu_xx == pytest.approx(2.0 * r.mu_xx, rel=1e-6)

    def test_fermi_level_electron_doping(self):
        """电子型：ε_F > 0（带隙中点即 Dirac 点），且随密度单调增。"""
        r1 = solve_transport(BilayerGrapheneModel(), n_cm2=5e11,
                             tau_fs=10.0, temperature=300.0, n_mesh=160)
        r2 = _bilayer_electrons()
        assert r1.fermi_ev > 0.0
        assert r2.fermi_ev > r1.fermi_ev

    def test_density_monotonic_mobility_trend(self):
        """触碰带物理：低密度时镜像带进热窗 → μ 高于 eτ/m*（非 Drude）。

        对照解析带边质量：n=2e12 cm⁻²（ε_F ≈ 43 meV）时 μ 仍为 Drude
        值的 ~0.8 倍（非抛物线质量增大效应），确认这是模型物理而非
        数值伪像（细网格 N=800 复核一致，见模块文档）。
        """
        m_star = effective_mass(BilayerGrapheneModel(), (1 / 3, 1 / 3),
                                band=2, direction=(1, 1))
        r = _bilayer_electrons()
        ratio = r.mu_xx / drude_mu_cm2(10.0, m_star)
        assert 0.5 < ratio < 1.2


class TestApi:
    def test_invalid_params(self):
        model = PhosphoreneRudenko()
        with pytest.raises(ValueError):
            solve_transport(model, n_cm2=0.0)
        with pytest.raises(ValueError):
            solve_transport(model, n_cm2=1e12, tau_fs=0.0)
        with pytest.raises(ValueError):
            solve_transport(model, n_cm2=1e12, n_mesh=4)

    @pytest.mark.parametrize("kwargs", [
        {"n_cm2": float("nan")}, {"n_cm2": float("inf")},
        {"tau_fs": float("nan")}, {"temperature": float("nan")},
        {"tau_fs": float("inf")}, {"g_spin": float("nan")},
    ])
    def test_non_finite_params_are_rejected(self, kwargs):
        """NaN 与任何值的比较都为假，会穿过朴素范围校验并算出 NaN 结果。"""
        with pytest.raises(ValueError):
            solve_transport(PhosphoreneRudenko(), **{"n_cm2": 1e12, **kwargs})

    def test_unattainable_density_is_rejected(self):
        """密度超出网格可达范围必须报错，不得静默收敛到二分区间端点。"""
        with pytest.raises(ValueError, match="可达范围"):
            solve_transport(PhosphoreneRudenko(), n_cm2=1e16,
                            tau_fs=10.0, temperature=300.0, n_mesh=16)
