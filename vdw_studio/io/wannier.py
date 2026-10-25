"""Wannier90 ``_hr.dat`` 的读写。

`_hr.dat` 是 Wannier90 输出的实空间哈密顿量：H(R) 在 Wannier 基下的
矩阵元，以 ``(R1 R2 R3 m n Re Im)`` 五元组逐行给出。本模块只做格式
转换；周期插值与模型封装见 ``vdw_studio.engine.wannier_model``。

格式（Wannier90 user guide）::

    <注释行>
    <nrpt> <ndegen>            # R 点数、简并因子个数
    <简并因子，每行至多 15 个>
    R1 R2 R3 m n Re Im × nrpt × nw × nw 行

约定：数据按 R 分块连续（Wannier90 的输出顺序）；块内 (m, n) 用显式
索引定位。单位能量 eV；R 以晶格矢（整数）为单位。H(R) 满足厄米配对
H(R)† = H(−R)——:func:`read_hr` 不做对称化（保持原始数据），对称性
由调用方检验。
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np


def read_hr(filepath) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """解析 ``_hr.dat``。

    返回 (r_vectors (nrpt,3) int, matrices (nrpt,nw,nw) complex,
    degeneracy (nrpt,) float, comment)。
    """
    path = Path(filepath)
    with path.open(encoding="utf-8") as stream:
        comment = stream.readline().strip()
        header = stream.readline().split()
        if len(header) != 2:
            raise ValueError(f"{path}:2: expected '<nrpt> <ndegen>' header")
        nrpt, ndegen_count = int(header[0]), int(header[1])
        if nrpt < 1 or ndegen_count < 1:
            raise ValueError(f"{path}:2: nrpt/ndegen must be positive")

        degeneracy: List[float] = []
        while len(degeneracy) < ndegen_count:
            line = stream.readline()
            if not line:
                raise ValueError(f"{path}: truncated degeneracy block "
                                 f"({len(degeneracy)}/{ndegen_count})")
            degeneracy.extend(float(token) for token in line.split())
        if any(d <= 0 for d in degeneracy[:nrpt]):
            raise ValueError(f"{path}: degeneracy factors must be positive")

        tuples = []
        for line_number, line in enumerate(stream, start=len(degeneracy) + 3):
            fields = line.split()
            if len(fields) != 7:
                raise ValueError(f"{path}:{line_number}: expected "
                                 f"'R1 R2 R3 m n Re Im', got {line.strip()!r}")
            try:
                tuples.append((int(fields[0]), int(fields[1]), int(fields[2]),
                               int(fields[3]) - 1, int(fields[4]) - 1,
                               complex(float(fields[5]), float(fields[6]))))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: malformed data row") from exc

    if not tuples:
        raise ValueError(f"{path}: no data rows")
    if len(tuples) % nrpt:
        raise ValueError(f"{path}: {len(tuples)} data rows not divisible by nrpt={nrpt}")
    rows_per_r = len(tuples) // nrpt
    nw = int(round(np.sqrt(rows_per_r)))
    if nw * nw != rows_per_r:
        raise ValueError(f"{path}: {rows_per_r} rows per R is not a perfect square")

    r_vectors = np.array([t[:3] for t in tuples[::rows_per_r]], dtype=int)
    matrices = np.zeros((nrpt, nw, nw), dtype=complex)
    for index, (r1, r2, r3, m, n, value) in enumerate(tuples):
        if not 0 <= m < nw or not 0 <= n < nw:
            raise ValueError(f"{path}: band index ({m + 1},{n + 1}) out of range for nw={nw}")
        r_index = index // rows_per_r
        if (r1, r2, r3) != tuple(r_vectors[r_index]):
            raise ValueError(f"{path}: data row {index} leaves its R block "
                             f"(expected {tuple(r_vectors[r_index])}, got {(r1, r2, r3)})")
        matrices[r_index, m, n] = value

    degeneracy_array = np.array(degeneracy[:nrpt], dtype=float)
    return r_vectors, matrices, degeneracy_array, comment


def write_hr(filepath, r_vectors, matrices, degeneracy, comment="Written by vdw_studio") -> None:
    """按 Wannier90 格式写出 ``_hr.dat``。

    ``r_vectors`` (nrpt,3) int；``matrices`` (nrpt,nw,nw) complex；
    ``degeneracy`` (nrpt,)（全 1 即可，插值时作除数）。
    """
    r_vectors = np.asarray(r_vectors, dtype=int)
    matrices = np.asarray(matrices, dtype=complex)
    degeneracy = np.asarray(degeneracy, dtype=float)
    if matrices.shape[0] != len(r_vectors) or len(degeneracy) != len(r_vectors):
        raise ValueError("r_vectors / matrices / degeneracy 长度不一致")
    nw = matrices.shape[1]
    lines = [comment, f"{len(r_vectors)} {len(degeneracy)}"]
    for start in range(0, len(degeneracy), 15):
        lines.append(" ".join(f"{v:g}" for v in degeneracy[start:start + 15]))
    for r_index, r in enumerate(r_vectors):
        for m in range(nw):
            for n in range(nw):
                value = matrices[r_index, m, n]
                lines.append(f"{r[0]:5d}{r[1]:5d}{r[2]:5d}{m + 1:5d}{n + 1:5d}"
                             f"  {value.real:18.12f}  {value.imag:18.12f}")
    Path(filepath).write_text("\n".join(lines) + "\n", encoding="utf-8")
