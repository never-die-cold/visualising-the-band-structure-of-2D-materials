"""CLI 命令行接口测试。"""

import json
from dataclasses import replace

import pytest
import numpy as np

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
        assert summary["gap"]["scope"] == "brillouin-zone"
        assert summary["gap"]["search"]["converged"] is True
        assert summary["path_gap"]["scope"] == "path"
        assert summary['task']['material_key'] == 'phosphorene'
        assert summary['task']['n_per_segment'] == 10
        assert summary['task']['dos_mesh'] == [12, 12]
        assert summary['task']['model_state']['state']['n_sites'] == 4
        assert summary['dos']['expected_states'] == 4
        assert summary['dos']['integral_in_window'] == pytest.approx(4, abs=1e-6)

    def test_run_kp_includes_masses(self, tmp_path):
        out = tmp_path / "mos2"
        assert cli.main(["run", "mos2_kp", "--out", str(out),
                         "--npoints", "10", "--mesh", "12"]) == 0
        summary = json.loads((out / "summary.json").read_text(
            encoding="utf-8"))
        masses = summary["effective_masses_m0"]
        assert 0.3 < masses["m_e_up"] < 0.6      # 文献 0.43-0.46
        assert 0.4 < masses["m_h_up"] < 0.8      # 文献 0.54-0.61
        assert summary["calculation_domain"] == "valley-local"
        assert summary["dos"]["available"] is False
        assert summary["gap"]["scope"] == "valley-local"
        assert summary["path_gap"] is None
        assert not (out / "dos.png").exists()
        with np.load(out / "bands.npz") as data:
            assert data["energies"].shape == (21, 4)
            assert len(data["ticks"]) == 3

    def test_run_unknown_material(self):
        with pytest.raises(ValueError):
            cli.main(["run", "NaCl", "--out", str(pytest.__file__)])

    def test_shifted_graphene_exports_separate_gap_domains(self, tmp_path, monkeypatch):
        from vdw_studio.engine.models import HoneycombModel
        from vdw_studio.engine.strain import apply_strain
        from vdw_studio.presets import PRESETS
        monkeypatch.setitem(PRESETS, "graphene", replace(
            PRESETS["graphene"], make_model=lambda: apply_strain(HoneycombModel(a=2.46), ex=.02)))
        out = tmp_path / "shifted_graphene"
        assert cli.main(["run", "graphene", "--out", str(out), "--npoints", "10",
                         "--mesh", "8", "--gap-mesh", "12"]) == 0
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        assert summary["gap"]["scope"] == "brillouin-zone"
        assert summary["gap"]["status"] == "zero-gap"
        assert summary["gap"]["gap_eV"] is None
        assert summary["gap"]["raw_gap_eV"] < 1e-7
        assert summary["gap"]["search"]["initial_mesh"] == [12, 12]
        assert summary["gap"]["search"]["refined_mesh"] == [24, 24]
        assert summary["gap"]["search"]["converged"] is True
        assert summary["path_gap"]["scope"] == "path"
        assert summary["path_gap"]["gap_eV"] > .1
        point = np.array(summary["gap"]["cbm_kfrac"])
        assert np.linalg.norm((point - [1/3, 1/3] + .5) % 1 - .5) > .001


class TestCliExport:
    def test_export_structure(self, tmp_path):
        out = tmp_path / "st"
        assert cli.main(["export", "MoS2", "--out", str(out)]) == 0
        assert (out / "MoS2.POSCAR").exists()
        assert (out / "MoS2.xyz").exists()


class TestCliBatch:
    def test_run_all_supported_presets(self, tmp_path):
        """实际计算与出图，不替换单材料入口。"""
        from matplotlib import pyplot as plt
        from vdw_studio.presets import get_preset, list_presets
        out = tmp_path / "batch"
        figures_before = plt.get_fignums()
        assert cli.main(["run-all", "--out", str(out),
                         "--npoints", "4", "--mesh", "4"]) == 0
        batch = json.loads((out / "batch_summary.json").read_text(encoding="utf-8"))
        supported = [key for key in list_presets()
                     if get_preset(key).engine != "sp3d5"]
        assert batch["completed"] == len(supported)
        assert batch["failed"] == 0 and batch["skipped"] == 1
        for key in supported:
            summary = json.loads((out / key / "summary.json").read_text(encoding="utf-8"))
            assert summary["material"] == key
            assert (out / key / "bands.png").exists()
        assert {r["material"] for r in batch["runs"]} == set(list_presets())
        assert plt.get_fignums() == figures_before

    def test_run_all_records_database(self, tmp_path, monkeypatch):
        from vdw_studio.storage import ResultsDB
        monkeypatch.setattr(cli, "list_presets", lambda: ["hbn", "mos2_kp"])
        db = tmp_path / "batch.db"
        assert cli.main(["run-all", "--out", str(tmp_path / "batch"),
                         "--npoints", "4", "--mesh", "4", "--db", str(db)]) == 0
        rows = ResultsDB(str(db)).history()
        assert {r["material"] for r in rows} == {"hbn", "mos2_kp"}
        assert len(rows) == 2

    def test_failed_material_does_not_stop_batch(self, tmp_path, monkeypatch):
        from vdw_studio.presets import PRESETS

        def fail_model():
            raise ValueError("模型构建测试错误")

        monkeypatch.setitem(PRESETS, "graphene", replace(
            PRESETS["graphene"], make_model=fail_model))
        monkeypatch.setattr(cli, "list_presets",
                            lambda: ["graphene", "hbn", "mos2_sp3d5"])
        out = tmp_path / "batch"
        assert cli.main(["run-all", "--out", str(out),
                         "--npoints", "4", "--mesh", "4"]) == 1
        batch = json.loads((out / "batch_summary.json").read_text(encoding="utf-8"))
        assert batch["completed"] == batch["failed"] == batch["skipped"] == 1
        failure = next(r for r in batch["runs"] if r["status"] == "failed")
        assert failure["material"] == "graphene"
        assert "模型构建测试错误" in failure["error"]
        assert (out / "hbn" / "summary.json").exists()
