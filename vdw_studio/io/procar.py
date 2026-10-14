"""VASP PROCAR 解析：投影能带（轨道/离子分辨）接口（Phase 6.3）。

PROCAR 是 VASP 输出的**投影系数**文件：每个 k 点、每条能带给出每个
离子的 s/p/d(/f) 轨道权重。vdW Studio 解析后可直接计算
「带 → 轨道字符」（例如 MoS₂ 价带顶的 Mo-d 主导、导带底的 d_z² 特征），
与 BandViz 的 EIGENVAL 能带图互补。

格式（VASP 官方文档，procar 节）：

    PROCAR file:  # of k-points:   58  # of bands:  12  # of ions:  3

     k-point    1 :   0.0  0.0  0.0   weight = 0.017241

    band    1 # energy  -60.12 # occ.  1.0000

    ion      s    py    pz    px   dxy   dyz   dz2   dxz dx2-y2   tot
      1  0.001 0.000 ...                              0.003
      2  ...
    tot  0.002 ...

- 首行 "PROCAR file:" 在旧版本中可能缺失——解析器容忍；
- 轨道列名从 ``ion`` 表头行**动态读取**（支持含 f 轨道的 16 列格式）；
- 每个 k 点/带的 ``tot`` 行单独保存（全系统轨道求和）。

本模块只做解析与权重计算，不做可视化；数据类可直接喂给
分析/绘图流程（详见 ``ProcarData.orbital_weights``）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

_KPOINT_RE = re.compile(r"k-point\s+\d+\s*:(?P<payload>.*)$")
_BAND_RE = re.compile(
    r"band\s+(\d+)\s*#\s*energy\s+([-\d.Ee+]+)\s*#\s*occ\.\s*([-\d.Ee+]+)")
_HEADER_RE = re.compile(
    r"#\s*of\s*k-points:\s*(\d+).*#\s*of\s*bands:\s*(\d+).*#\s*of\s*ions:\s*(\d+)",
    re.IGNORECASE)
_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass
class ProcarData:
    """PROCAR 投影能带数据。

    - ``energies``/``occupations``: (n_k, n_bands)；
    - ``projections``: (n_k, n_bands, n_ions, n_orbital) 逐离子轨道权重；
    - ``totals``: (n_k, n_bands, n_orbital) 全系统（tot 行）轨道求和。
    """

    n_kpoints: int
    n_bands: int
    n_ions: int
    orbital_names: list
    kpoints: np.ndarray
    weights: np.ndarray
    energies: np.ndarray
    occupations: np.ndarray
    projections: np.ndarray
    totals: np.ndarray

    # ------------------------------------------------------------------
    def orbital_weights(self, orbitals: Optional[Sequence[str]] = None,
                        ions: Optional[Sequence[int]] = None,
                        k_range: Optional[tuple] = None) -> np.ndarray:
        """所选轨道/离子的投影权重 (n_k, n_bands)。

        参数:
            orbitals: 轨道名列表（如 ["dxy", "dz2"]，见 ``orbital_names``）；
                None = 全部轨道。
            ions: 0 基离子序号列表；None = 全部离子。
            k_range: (start, stop) 切片（默认全部 k 点）。
        """
        orb_idx = self._orbital_indices(orbitals)
        ion_idx = (list(range(self.n_ions)) if ions is None
                   else [int(i) for i in ions])
        for i in ion_idx:
            if not 0 <= i < self.n_ions:
                raise ValueError(f"离子序号越界: {i}（0..{self.n_ions - 1}）")
        sl = slice(*(k_range or (None, None)))
        sub = self.projections[sl][:, :, ion_idx, :][:, :, :, orb_idx]
        return sub.sum(axis=(2, 3))

    def orbital_fraction(self, orbitals: Optional[Sequence[str]] = None,
                         ions: Optional[Sequence[int]] = None,
                         k_range: Optional[tuple] = None) -> np.ndarray:
        """所选轨道的归一化分数（除以全部轨道·全部离子总权重）(n_k, n_bands)。

        例：MoS₂ 价带顶的 Mo-d 分数 = orbital_fraction(d 轨道, ions=[0])。
        """
        sel = self.orbital_weights(orbitals, ions, k_range)
        total = self.projections.sum(axis=3).sum(axis=2)[slice(
            *(k_range or (None, None)))]
        return np.divide(sel, total, out=np.zeros_like(sel), where=total > 1e-12)

    def band_character(self, k_index: int, band_index: int,
                       top: int = 3) -> list:
        """某 k 点、某能带的主导轨道字符: [(轨道名, 归一化分数), ...] 降序。

        分数 = 该轨道（全离子求和）权重 / 全部轨道离子总权重。
        """
        block = self.projections[k_index, band_index]      # (n_ions, n_orb)
        per_orb = block.sum(axis=0)
        total = float(per_orb.sum())
        if total <= 1e-12:
            return []
        order = np.argsort(per_orb)[::-1][:top]
        return [(self.orbital_names[i], float(per_orb[i] / total))
                for i in order]

    def _orbital_indices(self, orbitals: Optional[Sequence[str]]) -> list:
        if orbitals is None:
            return list(range(len(self.orbital_names)))
        idx = []
        for name in orbitals:
            if name not in self.orbital_names:
                raise ValueError(
                    f"未知轨道 {name!r}；可用: {self.orbital_names}")
            idx.append(self.orbital_names.index(name))
        return idx


def parse_procar(text: str) -> ProcarData:
    """解析 PROCAR 文本（见模块文档的格式说明与容错规则）。"""
    lines = text.splitlines()
    header = None
    for ln in lines[:3]:
        m = _HEADER_RE.search(ln)
        if m:
            header = tuple(int(g) for g in m.groups())

    kpoints, weights = [], []
    energies, occs = [], []
    proj_rows: list = []          # 逐 (k, band) 的 (n_ions, n_orb) 权重
    tot_rows: list = []
    orbital_names: Optional[list] = None

    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        mk = _KPOINT_RE.search(line)
        if mk:
            vals = [float(v) for v in _FLOAT_RE.findall(mk.group("payload"))]
            if len(vals) < 4:
                raise ValueError(f"k-point 行数值不足: {line.strip()!r}")
            kpoints.append(vals[:3])
            weights.append(vals[-1])
            i += 1
            continue
        mb = _BAND_RE.search(line)
        if mb:
            energies.append(float(mb.group(2)))
            occs.append(float(mb.group(3)))
            # 找 ion 表头行
            j = i + 1
            while j < n and not lines[j].lstrip().lower().startswith("ion"):
                j += 1
            if j >= n:
                raise ValueError(
                    f"band 行之后未找到 ion 表头（行 {i + 1} 起）")
            names = [t for t in lines[j].split()
                     if t.lower() not in ("ion", "tot")]
            if not names:
                raise ValueError(f"ion 表头行无法解析轨道名（行 {j + 1}）")
            if orbital_names is None:
                orbital_names = names
            elif names != orbital_names:
                raise ValueError(
                    f"轨道列不一致：{orbital_names} vs {names}（行 {j + 1}）")
            n_orb = len(orbital_names)
            ion_block = np.zeros((header[2] if header else 0, n_orb))
            tot = np.zeros(n_orb)
            j += 1
            seen_ions = 0
            while j < n:
                stripped = lines[j].strip()
                if not stripped or stripped.startswith("band") \
                        or stripped.startswith("k-point") or _KPOINT_RE.search(lines[j]):
                    break
                if stripped.lower().startswith("tot"):
                    vals = _FLOAT_RE.findall(stripped)
                    if len(vals) < n_orb:
                        raise ValueError(f"tot 行数值不足（行 {j + 1}）")
                    tot = np.array([float(v) for v in vals[:n_orb]])
                    j += 1
                    break
                vals = _FLOAT_RE.findall(stripped)
                if len(vals) < n_orb + 1:
                    raise ValueError(
                        f"离子权重行数值不足（行 {j + 1}）: {stripped[:60]!r}")
                ion_no = int(float(vals[0]))
                if header and not 1 <= ion_no <= header[2]:
                    raise ValueError(f"离子序号越界（行 {j + 1}）: {ion_no}")
                while ion_block.shape[0] < ion_no:
                    ion_block = np.vstack([ion_block, np.zeros((1, n_orb))])
                ion_block[ion_no - 1] = [float(v) for v in vals[1:n_orb + 1]]
                seen_ions = max(seen_ions, ion_no)
                j += 1
            ion_block = ion_block[:seen_ions]
            proj_rows.append(ion_block)
            tot_rows.append(tot)
            i = j
            continue
        i += 1

    if not kpoints or not energies:
        raise ValueError("未解析到任何 k 点/能带数据（文件格式不符？）")
    n_k, n_b = len(kpoints), len(energies)
    if orbital_names is None:
        raise ValueError("未找到 ion 表头行，无法确定轨道列")
    if n_b % n_k != 0:
        raise ValueError(f"能带行数 {n_b} 不是 k 点数 {n_k} 的整数倍")
    n_bands = n_b // n_k
    n_orb = len(orbital_names)
    n_ions = max((r.shape[0] for r in proj_rows), default=0)
    if header and (n_k, n_bands, n_ions) != header:
        raise ValueError(
            f"数据与文件头不一致：解析 ({n_k}, {n_bands}, {n_ions}) "
            f"vs 头部 {header}")

    projections = np.zeros((n_k, n_bands, n_ions, n_orb))
    for idx, block in enumerate(proj_rows):
        ik, ib = divmod(idx, n_bands)
        projections[ik, ib, :block.shape[0], :] = block

    return ProcarData(
        n_kpoints=n_k, n_bands=n_bands, n_ions=n_ions,
        orbital_names=list(orbital_names),
        kpoints=np.array(kpoints),
        weights=np.array(weights),
        energies=np.array(energies).reshape(n_k, n_bands),
        occupations=np.array(occs).reshape(n_k, n_bands),
        projections=projections,
        totals=np.array(tot_rows).reshape(n_k, n_bands, n_orb),
    )


def read_procar(path: Union[str, Path]) -> ProcarData:
    """从文件读取 PROCAR。"""
    return parse_procar(Path(path).read_text(encoding="utf-8",
                                             errors="replace"))
