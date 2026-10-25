"""广义本征矢的（原子, 壳层）投影权重。

权重是"S-正交投影"的范数：广义本征问题 Hψ = E·Sψ 以 S 为度规，基矢
|a⟩ 彼此**不正交**（S_ab ≠ δ），因此子空间 A 上的权重必须含子空间重叠
矩阵的逆——记 c_A = (Sψ)_A（A 内基矢的 S-投影系数），则

    w_A = ⟨ψ|P_A|ψ⟩_S = c_A† (S_A)⁻¹ c_A，  P_A = Σ|a⟩(S_A⁻¹)_ab⟨b|S

完备组（A = 全部基函数）时 S_A = S，w = ψ†Sψ = 1。漏掉 S_A⁻¹ 会得到
‖Sψ‖² 那样的非归一量（本模块曾犯过这个错，扫描脚本与文档里的
"m=±2 权重 0.74" 即由此而来，已更正并标注待复核）。
"""

from typing import Dict, Optional, Tuple

import numpy as np
import scipy.linalg

# ZahidMoS2Model 的位点顺序：0=Mo，1/2=S；ORBITALS 每原子 9 轨道 × 2 自旋。
_ORBITALS = ("s", "px", "py", "pz", "dxy", "dyz", "dzx", "dx2-y2", "dz2")
_ORBITALS_PER_ATOM = 18

DEFAULT_GROUPS: Dict[str, Tuple[Tuple[int, ...], Tuple[str, ...]]] = {
    "Mo_s": ((0,), ("s",)),
    "Mo_p": ((0,), ("px", "py", "pz")),
    "Mo_d": ((0,), ("dxy", "dyz", "dzx", "dx2-y2", "dz2")),
    "S_s": ((1, 2), ("s",)),
    "S_p": ((1, 2), ("px", "py", "pz")),
    "S_d": ((1, 2), ("dxy", "dyz", "dzx", "dx2-y2", "dz2")),
}


def group_weights(model, kpoints, groups: Optional[dict] = None
                  ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """逐 k 点求解并返回 (energies, {组名: 权重 (nk, nbands)})。

    ``groups`` 缺省用 :data:`DEFAULT_GROUPS`；每组是 (原子索引元组, 轨道名
    元组)。一个组在某个原子上没有的轨道（如 S 的 d）会被静默跳过。
    """
    groups = DEFAULT_GROUPS if groups is None else groups
    kpoints = np.asarray(kpoints, dtype=float)
    if kpoints.ndim != 2 or kpoints.shape[1] not in (2, 3) or not len(kpoints):
        raise ValueError("kpoints must be a nonempty (nk, 2|3) array")
    nk = len(kpoints)
    nb = model.n_basis

    energies = np.empty((nk, nb))
    weights = {name: np.empty((nk, nb)) for name in groups}
    basis_indices: Dict[str, np.ndarray] = {}
    for name, (atoms, orbitals) in groups.items():
        indices = [atom * _ORBITALS_PER_ATOM + _ORBITALS.index(orbital) * 2 + spin
                   for atom in atoms for orbital in orbitals if orbital in _ORBITALS
                   for spin in (0, 1)]
        if not indices:
            raise ValueError(f"投影组 {name!r} 没有匹配到任何基函数")
        basis_indices[name] = np.array(indices)

    for ik, k in enumerate(kpoints):
        H, S = model.hamiltonian_overlap(k)
        w, v = scipy.linalg.eigh(H, S)
        energies[ik] = w
        s_psi = S @ v                          # (nb, nb)：逐列为 Sψ
        for name, indices in basis_indices.items():
            sub = s_psi[indices, :]            # c_A (n_A, nb)
            S_sub = S[np.ix_(indices, indices)]
            # w = c† S_A⁻¹ c（逐列）；S_A 厄米正定，直接求逆足够稳定
            S_A_inv = np.linalg.inv(S_sub)
            weights[name][ik] = np.einsum('ib,ij,jb->b', sub.conj(), S_A_inv, sub).real
    return energies, weights
