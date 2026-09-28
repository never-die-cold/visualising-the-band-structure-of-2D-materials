"""CLI 命令行接口测试。"""

import json

import pytest

from vdw_studio import cli


class TestCliList:
    def test_list_runs(self, capsys):
        assert cli.main(["list"]) == 0
        out = capsys.readouterr().out
        for key in ("graphene", "mos2_kp", "phosphorene", "mos2_sp3d5"):
            assert key in out

    def test_list_shows_references(self, capsys):
        cli.main(["list"])
        out = capsys.readouterr().out
        assert "1.670" in out      # MoS2 文献带隙
        assert "3.500" in out      # hBN


class TestCliRun:
    def test_run_phosphorene(self, tmp_path, capsys):
        out = tmp_path / "bp"
        rc = cli.main(["run", "phosphorene", "--out", str(out),
                       "--npoints", "10", "--mesh", "12"])
        assert rc == 0
        # 全部输出文件
        for name in ("structure.png", "bands.png", "dos.png", "bz.png",
                     "bands.npz", "dos.npz", "phosphorene.POSCAR",
                     "summary.json"):
            assert (out / name).exists(), f"缺少 {name}"
        summary = json.loads((out / "summary.json").read_text(
            encoding="utf-8"))
        assert summary["formula"] == "P4"
        assert summary["gap"]["gap_eV"] == pytest.approx(1.52, abs=0.02)
        assert summary["gap"]["matches_reference"] is True

    def test_run_kp_includes_masses(self, tmp_path):
        out = tmp_path / "mos2"
        assert cli.main(["run", "mos2_kp", "--out", str(out),
                         "--npoints", "10", "--mesh", "12"]) == 0
        summary = json.loads((out / "summary.json").read_text(
            encoding="utf-8"))
        masses = summary["effective_masses_m0"]
        assert 0.3 < masses["m_e_up"] < 0.6      # 文献 0.43-0.46
        assert 0.4 < masses["m_h_up"] < 0.8      # 文献 0.54-0.61

    def test_run_unknown_material(self):
        with pytest.raises(ValueError):
            cli.main(["run", "NaCl", "--out", str(pytest.__file__)])


class TestCliExport:
    def test_export_structure(self, tmp_path):
        out = tmp_path / "st"
        assert cli.main(["export", "MoS2", "--out", str(out)]) == 0
        assert (out / "MoS2.POSCAR").exists()
        assert (out / "MoS2.xyz").exists()
