"""Berry 曲率与谷选择光学定则（谷物理分析，Phase 4）。

实现标准公式（Xiao, Chang & Niu, Rev. Mod. Phys. 82, 1959 (2010) 的
现代极化表述，自含推导可验证）：

- **Berry 曲率**（速度算符形式）::

      Ω_n(k) = −2 Im Σ_{m≠n} ⟨n|v_x|m⟩⟨m|v_y|n⟩ / (E_n − E_m)²

  其中 v_i = ∂H/∂k_i（Å·eV，q 以 Å⁻¹ 计 → Ω 单位 Å²）。

- **谷 Chern 数**（半量子化）::

      C_valley = (1/2π) ∫_disk Ω d²q

  对纯大质量 Dirac 模型整个谷平面的通量为 ±π，即 C = ±1/2——
  谷-圆二色性与谷 Hall 效应的拓扑基础。

- **谷选择光学定则**（圆二色性）::

      F_±(q) = |⟨c| v_x ± i v_y |v⟩|²

  q→0 时单一圆偏振主导，且 K 与 −K 谷的选择相反
  （谷-光选择耦合，MoS₂ 等 TMD 谷光电子学的核心）。

适用对象：任何提供厄米矩阵 ``hamiltonian(q)`` 的模型
（TMDKpModel 及测试中的合成模型）。q 均为相对谷中心的
笛卡尔波矢 (Å⁻¹)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from ..numerics import finite_scalar, finite_vector, hermitian_matrix, integer


def _velocity_matrices(hamiltonian, q: np.ndarray, dq: float = 1e-7):
    """速度算符 (v_x, v_y) = ∂H/∂q_(x,y)（中心差分，eV·Å）。"""
    q = finite_vector(q, 2, 'q')
    dq = finite_scalar(dq, 'dq', positive=True)
    def H(dx, dy):
        return hermitian_matrix(hamiltonian(np.array([q[0] + dx, q[1] + dy])))
    Hx = (H(dq, 0) - H(-dq, 0)) / (2 * dq)
    Hy = (H(0, dq) - H(0, -dq)) / (2 * dq)
    return Hx, Hy


def _band_index(band, size, name='band'):
    band = integer(band, name, minimum=0)
    if band >= size:
        raise ValueError(f'{name} is outside the Hamiltonian basis')
    return band


def _check_isolated_band(energies, band):
    gaps = np.abs(energies - energies[band])
    gaps[band] = np.inf
    if np.min(gaps) <= 1e-10:
        raise ValueError('Single-band Berry/orbital analysis is undefined at degeneracy')


def berry_curvature(hamiltonian, q, band: int,
                    dq: float = 1e-6) -> float:
    """单 k 点、单带的 Berry 曲率 Ω (Å²)。

    参数:
        hamiltonian: callable(q 2-vector) → 厄米矩阵 (eV)。
        band: 按能量升序的带序号。
    """
    q = finite_vector(q, 2, 'q')
    H0 = hermitian_matrix(hamiltonian(q))
    band = _band_index(band, len(H0))
    Hx, Hy = _velocity_matrices(hamiltonian, q, dq)
    E, V = np.linalg.eigh(H0)
    _check_isolated_band(E, band)
    # 本征基下的速度矩阵
    vx = V.conj().T @ Hx @ V
    vy = V.conj().T @ Hy @ V
    n = band
    omega = 0.0 + 0.0j
    for m in range(len(E)):
        if m == n:
            continue
        omega += vx[n, m] * vy[m, n] / (E[n] - E[m]) ** 2
    return float(-2 * np.imag(omega))


def valley_chern(hamiltonian, band: int, qmax: float,
                 n_theta: int = 72, n_rad: int = 48,
                 valley: int = +1) -> Tuple[float, float]:
    """谷 disk 上的 Berry 通量积分 → 谷 Chern 贡献 C = (1/2π)∮Ω d²q。

    极坐标积分（同心圆环 + 梯形/环形权重）。

    返回:
        (C_half, flux) — C_half = flux/(2π)。
    """
    # 高斯–勒让德 × 均匀角度 采样
    qmax = finite_scalar(qmax, 'qmax', positive=True)
    n_theta = integer(n_theta, 'n_theta', minimum=3)
    n_rad = integer(n_rad, 'n_rad')
    if valley not in (-1, 1):
        raise ValueError('valley must be -1 or +1')
    xs, ws = np.polynomial.legendre.leggauss(n_rad)
    radii = qmax * (xs + 1) / 2
    wr = qmax * ws / 2
    thetas = np.arange(n_theta) * (2 * np.pi / n_theta)
    wt = 2 * np.pi / n_theta

    flux = 0.0
    for r, wr_ in zip(radii, wr):
        if r < 1e-9:
            continue
        for th in thetas:
            q = valley * r * np.array([np.cos(th), np.sin(th)])
            flux += wr_ * wt * berry_curvature(hamiltonian, q, band) * r
    c_half = flux / (2 * np.pi)
    return float(c_half), float(flux)


def valley_report(hamiltonian_K, hamiltonian_Kminus,
                  band_v: int, band_c: int,
                  qmax: float = 0.25, n_grid: int = 41, *, cancel_check=None) -> dict:
    """谷物理一站式报告：Berry 曲率热图 + 两谷圆二色性 + 价带 Ω 峰值。

    参数:
        hamiltonian_K: K 谷 (τ=+1) 的 callable(q 2-vector) → 厄米矩阵。
        hamiltonian_Kminus: −K 谷 (τ=−1) 的同签名 callable。
        band_v / band_c: 价带/导带序号（能量升序）。
        qmax: 热图半宽 (Å⁻¹)。
        n_grid: 热图每边采样数。

    返回:
        dict(q=qs, omega=Ω 网格 (n,n), omega_vb_K=价带 Ω(K),
             dichroism_K, dichroism_Kminus)
    """
    qmax = finite_scalar(qmax, 'qmax', positive=True)
    n_grid = integer(n_grid, 'n_grid', minimum=3)
    if n_grid % 2 == 0:
        raise ValueError('n_grid must be odd to sample the valley center')
    qs = np.linspace(-qmax, qmax, n_grid)
    omega = np.zeros((n_grid, n_grid))
    for i, qx in enumerate(qs):
        if cancel_check is not None:
            cancel_check()
        for j, qy in enumerate(qs):
            omega[i, j] = berry_curvature(hamiltonian_K,
                                          np.array([qx, qy]), band=band_v)
    d_k = optical_circular_dichroism(hamiltonian_K, np.array([5e-3, 0.0]),
                                     band_v=band_v, band_c=band_c)
    d_km = optical_circular_dichroism(hamiltonian_Kminus,
                                      np.array([5e-3, 0.0]),
                                      band_v=band_v, band_c=band_c)
    return {
        "q": qs,
        "omega": omega,                    # 行 = qy (绘图时转置)
        "omega_vb_K": float(omega[n_grid // 2, n_grid // 2]),
        "dichroism_K": d_k,
        "dichroism_Kminus": d_km,
    }


@dataclass
class DichroismResult:
    """谷选择光学跃迁结果。"""

    f_plus: float        # σ⁺ 振子强度 |⟨c|vx+i vy|v⟩|²
    f_minus: float       # σ⁻ 振子强度
    ratio: float         # 主导/次主导之比 (>1)
    dominant: str        # "sigma+" 或 "sigma-"


def optical_circular_dichroism(hamiltonian, q, band_v: int, band_c: int,
                               dq: float = 1e-6) -> DichroismResult:
    """谷选择圆二色性：σ± 光学跃迁强度（谷光电子学核心定则）。

    F_± = |⟨c| v_x ± i v_y |v⟩|²——q→0 时单一圆偏振主导，
    且 K (τ=+1) 与 −K (τ=−1) 谷的选择相反。
    """
    q = finite_vector(q, 2, 'q')
    H0 = hermitian_matrix(hamiltonian(q))
    band_v = _band_index(band_v, len(H0), 'band_v')
    band_c = _band_index(band_c, len(H0), 'band_c')
    if band_v >= band_c:
        raise ValueError('band_v must be below band_c')
    Hx, Hy = _velocity_matrices(hamiltonian, q, dq)
    E, V = np.linalg.eigh(H0)
    vx = V.conj().T @ Hx @ V
    vy = V.conj().T @ Hy @ V
    m_plus = vx[band_c, band_v] + 1j * vy[band_c, band_v]
    m_minus = vx[band_c, band_v] - 1j * vy[band_c, band_v]
    f_plus = float(abs(m_plus) ** 2)
    f_minus = float(abs(m_minus) ** 2)
    if f_plus >= f_minus:
        return DichroismResult(f_plus, f_minus,
                               f_plus / max(f_minus, 1e-30), "sigma+")
    return DichroismResult(f_plus, f_minus,
                           f_minus / max(f_plus, 1e-30), "sigma-")
