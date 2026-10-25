"""Wannier90 互操作回归测试。

验证锚点是**合成往返**：把平台自己的位点 TB 模型（石墨烯）导出成
Wannier90 ``_hr.dat`` 格式，再用 :class:`WannierModel` 读回插值——两条
完全独立的实现，能带逐位一致即证明格式解析与周期插值都正确。
真实 Wannier90 输出（官方示例）的对账待真实数据到位后补。
"""

import numpy as np
import pytest

from vdw_studio.engine.kpath import KPath
from vdw_studio.engine.models import HoneycombModel
from vdw_studio.engine.solver import solve_bands, solve_dos
from vdw_studio.engine.wannier_model import WannierModel
from vdw_studio.analysis import analyze_gap
from vdw_studio.io.wannier import read_hr, write_hr


def model_to_hr_entries(model, taus):
    """位点 TB 模型的 hopping 表 → (R 列表, [H(R) 矩阵])，厄米配对补全。

    Wannier 中心跟随原子：hopping 的分数位移 dfrac 对应实空间格矢
    ``R = dfrac + τ_i − τ_j``（τ 为各位点的分数坐标）。模型装配约定
    H(k)_ij = Σ t·e^{i k·dfrac}（自动补共轭）映射为
    H(R)[i,j] = t 与厄米配对 H(−R)[j,i] = t*。
    """
    taus = np.asarray(taus, dtype=float)
    nw = model.n_sites
    entries = {}
    for i, energy in enumerate(model.onsite):
        entries[(0, 0, 0, i, i)] = complex(energy)
    for h in model.hoppings:
        r = tuple(int(x) for x in np.rint(
            np.asarray(h.dfrac) + taus[h.i] - taus[h.j]))
        entries[(r + (h.i, h.j))] = complex(h.t)
    paired = dict(entries)
    for (rx, ry, rz, i, j), value in entries.items():
        paired.setdefault((-rx, -ry, -rz, j, i), np.conj(value))
    r_vectors = sorted({key[:3] for key in paired})
    matrices = {r: np.zeros((nw, nw), dtype=complex) for r in r_vectors}
    for (rx, ry, rz, i, j), value in paired.items():
        matrices[(rx, ry, rz)][i, j] += value
    return [np.array(r) for r in r_vectors], [matrices[r] for r in r_vectors]


@pytest.fixture
def honeycomb_hr(tmp_path):
    model = HoneycombModel(a=2.46, t=-2.7)
    taus = np.array([[0.0, 0.0, 0.0], [2 / 3, 1 / 3, 0.0]])   # A/B 子格中心（含 z）
    r_vectors, matrices = model_to_hr_entries(model, taus)
    path = tmp_path / "graphene_hr.dat"
    write_hr(path, r_vectors, matrices, np.ones(len(r_vectors)))
    return model, path


class TestHrFormat:
    def test_write_read_roundtrip(self, honeycomb_hr, tmp_path):
        model, path = honeycomb_hr
        r_vectors, matrices, degeneracy, comment = read_hr(path)
        assert len(r_vectors) == 5          # onsite(R=0) + 3 近邻 + 厄米配对引入的 ±R
        assert comment == "Written by vdw_studio"
        assert matrices.shape == (5, 2, 2)
        assert (degeneracy == 1).all()
        # 再写出再读入：字节级语义一致
        second = tmp_path / "again.dat"
        write_hr(second, r_vectors, matrices, degeneracy, comment="second")
        r2, m2, d2, c2 = read_hr(second)
        np.testing.assert_array_equal(r2, r_vectors)
        np.testing.assert_allclose(m2, matrices)
        assert c2 == "second"

    def test_truncated_file_rejected(self, tmp_path):
        bad = tmp_path / "bad.dat"
        # nrpt=2（需 2×nw² 行）却只给 1 行合法数据 → 行数校验必须拦住
        bad.write_text("x\n2 1\n1\n1 1 0 1 1 0.0 0.0\n")
        with pytest.raises(ValueError, match="not divisible"):
            read_hr(bad)


class TestWannierModel:
    def test_bands_match_original_model(self, honeycomb_hr):
        model, path = honeycomb_hr
        wannier = WannierModel(path, lattice=model.lattice, n_valence=1)
        assert wannier.n_sites == 2
        kpoints = np.array([[0.0, 0.0], [1 / 3, 1 / 3], [0.2, 0.1], [0.5, 0.0]])
        np.testing.assert_allclose(wannier.bands(kpoints), model.bands(kpoints),
                                   atol=1e-12)

    def test_hermiticity_and_valence(self, honeycomb_hr):
        model, path = honeycomb_hr
        wannier = WannierModel(path, lattice=model.lattice, n_valence=1)
        for k in ([0.0, 0.0], [0.13, 0.27], [1 / 3, 1 / 3]):
            h = wannier.hamiltonian(k)
            assert np.abs(h - h.conj().T).max() < 1e-13

    def test_feeds_solver_gap_and_dos(self, honeycomb_hr):
        model, path = honeycomb_hr
        wannier = WannierModel(path, lattice=model.lattice, n_valence=1)
        gap = analyze_gap(wannier, n_valence=1, mesh=(12, 12))
        assert gap.status == "zero-gap" and gap.raw_gap < 1e-7
        bs = solve_bands(wannier, KPath.for_lattice(model.lattice), n_per_segment=10)
        assert bs.n_valence == 1
        assert np.abs(bs.energies).min() == pytest.approx(0.0, abs=1e-8)
        dos = solve_dos(wannier, mesh=(8, 8), n_points=101)
        assert np.isfinite(dos.dos).all()
        assert dos.expected_states == pytest.approx(2.0)   # 2 条 Wannier 带
