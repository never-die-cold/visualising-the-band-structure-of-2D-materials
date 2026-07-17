import numpy as np
from scipy.optimize import curve_fit
from typing import Optional, Tuple
from .parser import BandData


class BandAnalyzer:
    """能带分析工具"""

    # ℏ² / (2m₀)  ≈ 3.80998 eV·Å²
    HBAR2_OVER_2M0 = 3.80998

    def __init__(
        self,
        band_data: BandData,
        lattice_constant_angstrom: float = 1.0
    ):
        """
        参数:
            band_data: 解析后的能带数据
            lattice_constant_angstrom: 2D 材料面内晶格常数（Å），
                                     用于把 k 点从分数坐标转换为 Å⁻¹
        """
        self.data = band_data
        self.fermi_level = 0.0
        self.lattice_constant_angstrom = lattice_constant_angstrom

    def set_fermi_level(self, efermi: float):
        """设置费米能级"""
        self.fermi_level = efermi

    def get_band_gap(self) -> dict:
        """
        计算带隙，返回包含 gap、direct、vbm、cbm 的字典。
        direct 判断基于 VBM 和 CBM 是否在同一 k 点索引。
        """
        energies = self.data.energies - self.fermi_level
        nk, nb = energies.shape

        if nk == 0 or nb == 0:
            return {'gap': None, 'direct': None, 'vbm': None, 'cbm': None}

        # 分别记录每个 k 点的最高占据态和最低未占据态及其索引
        occupied_info = []   # (energy, k_index)
        unoccupied_info = [] # (energy, k_index)

        for ik in range(nk):
            e_k = energies[ik]
            occ = e_k[e_k <= 0]
            unocc = e_k[e_k > 0]

            if len(occ) > 0:
                idx = int(np.argmax(occ))
                occupied_info.append((float(occ[idx]), ik))
            if len(unocc) > 0:
                idx = int(np.argmin(unocc))
                unoccupied_info.append((float(unocc[idx]), ik))

        if not occupied_info or not unoccupied_info:
            return {'gap': None, 'direct': None, 'vbm': None, 'cbm': None}

        # VBM = 全局最高占据态；CBM = 全局最低未占据态
        vbm, vbm_k = max(occupied_info, key=lambda x: x[0])
        cbm, cbm_k = min(unoccupied_info, key=lambda x: x[0])
        gap = cbm - vbm

        direct = (vbm_k == cbm_k)

        return {
            'gap': round(gap, 4),
            'direct': direct,
            'vbm': round(vbm, 4),
            'cbm': round(cbm, 4),
            'vbm_k': vbm_k,
            'cbm_k': cbm_k,
        }

    def effective_mass(
        self,
        band_index: int,
        k_index: int,
        num_points: int = 5
    ) -> Optional[float]:
        """
        在指定能带和 k 点附近抛物线拟合，计算有效质量（单位：m₀）。

        参数:
            band_index: 能带索引
            k_index: 参考 k 点索引
            num_points: 拟合时从 k_index 向两侧取的点数
        """
        if not (0 <= band_index < self.data.num_bands):
            return None
        if not (0 <= k_index < len(self.data.kdistances)):
            return None

        energies = self.data.energies[:, band_index] - self.fermi_level
        kdist_frac = self.data.kdistances

        # 把分数坐标的 k 距离转换为 Å⁻¹
        # 对 2D 材料近似：k(Å⁻¹) = k(frac) * 2π / a
        scale = 2.0 * np.pi / self.lattice_constant_angstrom
        kdist_angstrom = kdist_frac * scale

        start = max(0, k_index - num_points)
        end = min(len(kdist_angstrom), k_index + num_points + 1)

        k_local = kdist_angstrom[start:end]
        e_local = energies[start:end]

        if len(k_local) < 3:
            return None

        # 避免线性或噪声数据影响，要求数据有曲率
        if np.std(e_local) < 1e-6:
            return None

        k0 = kdist_angstrom[k_index]

        def parabola(k, a, b):
            return a * (k - k0) ** 2 + b

        try:
            popt, _ = curve_fit(parabola, k_local, e_local)
            a = popt[0]  # eV·Å²

            if abs(a) < 1e-10:
                return None

            # E = a (k - k0)² + b  ↔  E = E0 ± (ℏ² / 2m*) (k - k0)²
            # a = ± ℏ² / (2m*)  ⇒  m* = ℏ² / (2a) = 3.80998 / a
            m_star = self.HBAR2_OVER_2M0 / a
            return round(m_star, 4)
        except Exception:
            return None

        return None

    def estimate_effective_masses_at_gap(self) -> dict:
        """
        在 VBM 和 CBM 附近分别估算有效质量。
        返回 {vbm_mass, cbm_mass}，单位 m₀。
        """
        gap_info = self.get_band_gap()
        if gap_info['gap'] is None:
            return {'vbm_mass': None, 'cbm_mass': None}

        vbm_k = gap_info['vbm_k']
        cbm_k = gap_info['cbm_k']

        # 默认取价带顶上方第一条能带、导带底下方第一条能带
        # 这里简化为 num_electrons 决定填充的能带数
        num_electrons = self.data.num_electrons
        # 假设非自旋极化，每个能带可填充 2 电子
        occupied_band = num_electrons // 2 - 1
        unoccupied_band = occupied_band + 1

        vbm_mass = self.effective_mass(occupied_band, vbm_k)
        cbm_mass = self.effective_mass(unoccupied_band, cbm_k)

        return {
            'vbm_mass': vbm_mass,
            'cbm_mass': cbm_mass,
        }
