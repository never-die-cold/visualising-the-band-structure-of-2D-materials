import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DosData:
    """态密度数据结构"""
    energies: np.ndarray       # 能量网格 (ne,)
    total_dos: np.ndarray      # 总态密度 (ne,)
    vb_dos: Optional[np.ndarray] = None   # 价带态密度 (ne,)
    cb_dos: Optional[np.ndarray] = None   # 导带态密度 (ne,)
    fermi_level: float = 0.0
    energy_range: Tuple[float, float] = (-10.0, 10.0)
    sigma: float = 0.05        # 高斯展宽宽度 (eV)


class DosAnalyzer:
    """态密度分析工具：从能带数据（EIGENVAL）计算 DOS"""

    def __init__(self, energies: np.ndarray, fermi_level: float = 0.0):
        """
        参数:
            energies: (nk, nbands) 能带能量数组
            fermi_level: 费米能级 (eV)
        """
        self.energies = energies
        self.fermi_level = fermi_level
        self._dos_data: Optional[DosData] = None

    def set_fermi_level(self, efermi: float):
        """设置费米能级并清除缓存"""
        self.fermi_level = efermi
        self._dos_data = None

    def calculate_dos(
        self,
        energy_range: Tuple[float, float] = (-10.0, 10.0),
        num_points: int = 1000,
        sigma: float = 0.05,
        per_kpoint_weight: Optional[np.ndarray] = None
    ) -> DosData:
        """
        使用高斯展宽从能带数据计算态密度。

        参数:
            energy_range: 能量范围 (eV，相对于费米能级)
            num_points: 能量网格点数
            sigma: 高斯展宽标准差 (eV)
            per_kpoint_weight: 每个 k 点的权重，None 则均匀权重
        """
        e_min, e_max = energy_range
        energy_grid = np.linspace(e_min, e_max, num_points)

        # 所有能带能量展平
        all_energies = self.energies.flatten() - self.fermi_level

        if per_kpoint_weight is not None:
            # 每个 k 点有不同权重（例如 k 点积分权重）
            weights = np.repeat(per_kpoint_weight, self.energies.shape[1])
            total_dos = self._gaussian_broadening(
                all_energies, energy_grid, sigma, weights
            )
        else:
            total_dos = self._gaussian_broadening(all_energies, energy_grid, sigma)

        # 分别计算价带和导带 DOS
        vb_mask = all_energies <= 0
        cb_mask = all_energies > 0

        vb_dos = self._gaussian_broadening(
            all_energies[vb_mask], energy_grid, sigma
        ) if np.any(vb_mask) else np.zeros_like(energy_grid)

        cb_dos = self._gaussian_broadening(
            all_energies[cb_mask], energy_grid, sigma
        ) if np.any(cb_mask) else np.zeros_like(energy_grid)

        # 归一化：让总 DOS 的积分大致合理
        # 对于离散 k 点采样，需要乘以 k 点权重因子
        # 这里做简单归一化：总态密度峰值归一到合理范围
        if np.max(total_dos) > 0:
            # 保持原始比例，仅做数值稳定性处理
            total_dos = np.where(total_dos < 1e-10, 0, total_dos)

        self._dos_data = DosData(
            energies=energy_grid,
            total_dos=total_dos,
            vb_dos=vb_dos,
            cb_dos=cb_dos,
            fermi_level=self.fermi_level,
            energy_range=energy_range,
            sigma=sigma
        )
        return self._dos_data

    @staticmethod
    def _gaussian_broadening(
        eigenvalues: np.ndarray,
        energy_grid: np.ndarray,
        sigma: float,
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        高斯展宽计算态密度。

        DOS(E) = Σ_i w_i * exp(-(E - ε_i)² / (2σ²)) / (σ * √(2π))
        """
        if weights is None:
            weights = np.ones_like(eigenvalues)

        # 向量化计算： (ne, 1) - (1, nb) -> (ne, nb)
        diff = energy_grid[:, np.newaxis] - eigenvalues[np.newaxis, :]
        gauss = np.exp(-0.5 * (diff / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
        dos = np.sum(gauss * weights[np.newaxis, :], axis=1)
        return dos

    def get_dos_at_fermi(self) -> float:
        """获取费米能级处的态密度"""
        if self._dos_data is None:
            self.calculate_dos()
        # 在能量网格中找到最接近 0 的点
        idx = np.argmin(np.abs(self._dos_data.energies))
        return float(self._dos_data.total_dos[idx])

    def get_band_edges_from_dos(
        self,
        threshold: float = 0.01
    ) -> dict:
        """
        从 DOS 推断带边位置。
        价带顶 = DOS 从 0 开始显著上升的最后一个能量点（<0）
        导带底 = DOS 从 0 开始显著上升的第一个能量点（>0）
        """
        if self._dos_data is None:
            self.calculate_dos()

        energies = self._dos_data.energies
        dos = self._dos_data.total_dos

        # 找到 DOS 超过阈值的区域
        mask = dos > threshold * np.max(dos)

        vb_indices = np.where(mask & (energies <= 0))[0]
        cb_indices = np.where(mask & (energies > 0))[0]

        vbm = float(energies[vb_indices[-1]]) if len(vb_indices) > 0 else None
        cbm = float(energies[cb_indices[0]]) if len(cb_indices) > 0 else None
        gap = round(cbm - vbm, 4) if (vbm is not None and cbm is not None) else None

        return {
            'vbm': vbm,
            'cbm': cbm,
            'gap': gap
        }


class DoscarParser:
    """解析 VASP DOSCAR 文件"""

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)

    def parse(self) -> Optional[DosData]:
        """解析 DOSCAR 文件，返回总态密度数据"""
        if not self.filepath.exists():
            return None

        with open(self.filepath, 'r') as f:
            lines = f.readlines()

        if len(lines) < 6:
            return None

        # DOSCAR 第 6 行：EMIN, EMAX, NEDOS, EFERMI, weight
        try:
            header = lines[5].strip().split()
            emin = float(header[0])
            emax = float(header[1])
            nedos = int(header[2])
            efermi = float(header[3])
        except (ValueError, IndexError):
            return None

        # 第 7 行开始是总 DOS 数据
        data_start = 6
        energies = []
        total_dos = []

        for i in range(data_start, min(data_start + nedos, len(lines))):
            parts = lines[i].strip().split()
            if len(parts) >= 2:
                energies.append(float(parts[0]))
                total_dos.append(float(parts[1]))

        if len(energies) == 0:
            return None

        energies = np.array(energies) - efermi  # 对齐到费米能级
        total_dos = np.array(total_dos)

        # 分离价带和导带
        vb_mask = energies <= 0
        cb_mask = energies > 0

        vb_dos = np.where(vb_mask, total_dos, 0)
        cb_dos = np.where(cb_mask, total_dos, 0)

        return DosData(
            energies=energies,
            total_dos=total_dos,
            vb_dos=vb_dos,
            cb_dos=cb_dos,
            fermi_level=efermi,
            energy_range=(float(np.min(energies)), float(np.max(energies))),
            sigma=0.0  # DOSCAR 已是展宽后的结果
        )
