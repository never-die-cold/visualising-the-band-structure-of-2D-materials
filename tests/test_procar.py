"""VASP PROCAR 解析测试（Phase 6.3 投影能带接口）。

用合成 fixture（MoS₂ 型 3 离子 / 9 轨道的 s-p-d 格式）验证：
- 头部一致性、k 点/能量/占据/权重解析；
- 投影张量形状与逐值正确（手工构造的可验数值）；
- 轨道/离子选择求和、归一分数的正确性；
- band_character 的主导轨道识别（价带顶 Mo-d_z² 型主导）；
- 缺首行头（旧版 VASP）与畸形文件（数值不足/轨道列不一致）的容错。
"""

import numpy as np
import pytest

from vdw_studio.io.procar import parse_procar, read_procar

ORBITALS = ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "dx2-y2"]


def _make_procar(n_k=2, include_header=True):
    """构造 MoS₂ 型 PROCAR：2 离子 Mo（1, 2）+ S，2 band。

    band 0（价带顶）：Mo 的 dz2/dxy 主导（分数可手算）；
    band 1（导带底）：Mo 的 dz2 主导。
    """
    L = []
    if include_header:
        L.append("PROCAR file:      # of k-points:   2         "
                 "# of bands:   2         # of ions:   3")
    kpts = [(0.0, 0.0, 0.0, 0.5), (0.3333, 0.3333, 0.0, 0.5)]
    # 权重表: (k, band) -> [离子0 的 9 列, 离子1, 离子2]
    w = {
        (0, 0): [[0, 0, 0.05, 0, 0.30, 0, 0.45, 0, 0.10],
                 [0.02] + [0] * 8, [0.03] + [0] * 8],
        (0, 1): [[0, 0, 0.02, 0, 0.05, 0, 0.80, 0, 0.03],
                 [0.04] + [0] * 8, [0.04] + [0] * 8],
        (1, 0): [[0, 0, 0.06, 0, 0.25, 0, 0.50, 0, 0.08],
                 [0.03] + [0] * 8, [0.02] + [0] * 8],
        (1, 1): [[0, 0, 0.03, 0, 0.04, 0, 0.78, 0, 0.02],
                 [0.05] + [0] * 8, [0.03] + [0] * 8],
    }
    energies = {(0, 0): -1.20, (0, 1): 0.60, (1, 0): -0.90, (1, 1): 0.95}
    for ik, (kx, ky, kz, wt) in enumerate(kpts):
        L.append("")
        L.append(f" k-point {ik + 1:4d} :   {kx:11.8f}{ky:11.8f}"
                 f"{kz:11.8f}     weight = {wt:.6f}")
        for ib in range(2):
            L.append("")
            L.append(f"band {ib + 1:4d} # energy {energies[(ik, ib)]:11.8f}"
                     f" # occ.  {1.0 if ib == 0 else 0.0:.8f}")
            L.append("")
            L.append("ion      " + "".join(f"{o:>8}" for o in ORBITALS)
                     + "     tot")
            colsum = np.zeros(9)
            for ion in range(3):
                row = w[(ik, ib)][ion]
                colsum += np.array(row)
                L.append(f"{ion + 1:4d} " + "".join(f"{v:8.3f}" for v in row)
                         + f"{sum(row):9.3f}")
            L.append("tot  " + "".join(f"{v:8.3f}" for v in colsum)
                     + f"{colsum.sum():9.3f}")
    return "\n".join(L) + "\n"


class TestParsing:
    def test_shapes_and_values(self):
        d = parse_procar(_make_procar())
        assert (d.n_kpoints, d.n_bands, d.n_ions) == (2, 2, 3)
        assert d.orbital_names == ORBITALS
        assert d.kpoints.shape == (2, 3)
        assert np.allclose(d.weights, 0.5)
        assert d.energies[0, 0] == pytest.approx(-1.20)
        assert d.energies[1, 1] == pytest.approx(0.95)
        assert d.occupations[0, 1] == 0.0
        assert d.projections.shape == (2, 2, 3, 9)
        # 手工核对: k=0, band=0, 离子 0 的 dz2 = 0.45
        assert d.projections[0, 0, 0, 6] == pytest.approx(0.45)
        assert d.projections[0, 0, 2, 0] == pytest.approx(0.03)   # S 的 s

    def test_totals_row(self):
        """tot 行 = 逐离子逐轨道求和。"""
        d = parse_procar(_make_procar())
        assert np.allclose(d.totals, d.projections.sum(axis=2))

    def test_header_optional(self):
        """旧版 VASP 缺 'PROCAR file:' 首行也应解析。"""
        d = parse_procar(_make_procar(include_header=False))
        assert (d.n_kpoints, d.n_bands, d.n_ions) == (2, 2, 3)

    def test_read_from_file(self, tmp_path):
        p = tmp_path / "PROCAR"
        p.write_text(_make_procar(), encoding="utf-8")
        d = read_procar(p)
        assert d.n_bands == 2


class TestWeights:
    def test_orbital_selection(self):
        """d 轨道（5 列）求和 = 手算值。"""
        d = parse_procar(_make_procar())
        d_orb = ["dxy", "dyz", "dz2", "dxz", "dx2-y2"]
        w = d.orbital_weights(d_orb, ions=[0])
        # k=0, band=0: 0.30+0.45+0.10 = 0.85
        assert w[0, 0] == pytest.approx(0.85)
        # 全部离子：0.85 + (离子1+2 的 d 都为 0) = 0.85
        assert d.orbital_weights(d_orb)[0, 0] == pytest.approx(0.85)

    def test_orbital_fraction_normalized(self):
        """Mo-d 分数（离子 0）: 0.85 / 总权重。"""
        d = parse_procar(_make_procar())
        d_orb = ["dxy", "dyz", "dz2", "dxz", "dx2-y2"]
        frac = d.orbital_fraction(d_orb, ions=[0])
        total = 0.05 + 0.30 + 0.45 + 0.10 + 0.02 + 0.03      # k0b0 全部
        assert frac[0, 0] == pytest.approx(0.85 / total)
        assert 0.0 <= frac.min() and frac.max() <= 1.0

    def test_band_character_dominant(self):
        """价带顶主导轨道 = Mo 的 dz2（fixture 设计值）。"""
        d = parse_procar(_make_procar())
        chars = d.band_character(k_index=0, band_index=0, top=2)
        assert chars[0][0] == "dz2"
        assert chars[0][1] == pytest.approx(0.45 / d.projections[0, 0].sum(),
                                            rel=1e-6)
        # 导带底同样 dz2 主导，且分数更高（0.80/0.98）
        chars_c = d.band_character(k_index=0, band_index=1, top=1)
        assert chars_c[0][0] == "dz2"
        assert chars_c[0][1] > 0.75

    def test_invalid_orbital_name(self):
        d = parse_procar(_make_procar())
        with pytest.raises(ValueError):
            d.orbital_weights(["fz3"])
        with pytest.raises(ValueError):
            d.orbital_weights(ions=[5])


class TestErrors:
    def test_missing_ion_header(self):
        text = _make_procar().replace("ion      ", "xyz      ")
        with pytest.raises(ValueError):
            parse_procar(text)

    def test_inconsistent_orbital_columns(self):
        text = _make_procar()
        # 把第二个 band 的表头改动一列名
        idx = text.find("ion      ", text.find("band    2"))
        text = (text[:idx] + text[idx:].replace("dx2-y2", "dx2y2", 1))
        with pytest.raises(ValueError):
            parse_procar(text)

    def test_truncated_values(self):
        lines = _make_procar().splitlines()
        for i, ln in enumerate(lines):
            if ln.startswith("   2 "):
                lines[i] = "   2  0.02 0.0"
                break
        with pytest.raises(ValueError):
            parse_procar("\n".join(lines))

    def test_garbage_file(self):
        with pytest.raises(ValueError):
            parse_procar("this is not a procar file\n")
