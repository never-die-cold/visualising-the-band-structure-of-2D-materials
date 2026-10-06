"""结果数据库测试（SQLite 往返 + CLI 集成）。"""

import pytest

from vdw_studio import cli
from vdw_studio.storage import ResultsDB


class TestResultsDB:
    def test_record_and_history_roundtrip(self, tmp_path):
        db = ResultsDB(str(tmp_path / "t.db"))
        rid = db.record(material="mos2_kp", engine="kp", formula="MoS2",
                        gap_eV=1.67, gap_direct=True,
                        payload={"m_e": 0.44})
        rows = db.history()
        assert len(rows) == 1 and rows[0]["id"] == rid
        assert rows[0]["material"] == "mos2_kp"
        assert rows[0]["gap_eV"] == pytest.approx(1.67)
        assert rows[0]["gap_direct"] is True
        assert rows[0]["payload"]["m_e"] == 0.44

    def test_material_filter_and_order(self, tmp_path):
        db = ResultsDB(str(tmp_path / "t.db"))
        db.record(material="graphene", engine="tb", gap_eV=None)
        db.record(material="mos2_kp", engine="kp", gap_eV=1.67,
                  gap_direct=True)
        db.record(material="mos2_kp", engine="kp", gap_eV=1.67,
                  gap_direct=True)
        rows = db.history(material="mos2_kp")
        assert len(rows) == 2
        assert rows[0]["id"] > rows[1]["id"]      # 时间倒序
        rows_all = db.history()
        assert len(rows_all) == 3

    def test_null_gap(self, tmp_path):
        """零带隙材料 gap_eV=None 往返。"""
        db = ResultsDB(str(tmp_path / "t.db"))
        db.record(material="graphene", engine="tb", gap_eV=None,
                  gap_direct=None)
        row = db.history()[0]
        assert row["gap_eV"] is None and row["gap_direct"] is None


class TestCliIntegration:
    def test_run_with_db(self, tmp_path):
        out = tmp_path / "run"
        dbp = tmp_path / "res.db"
        assert cli.main(["run", "mos2_kp", "--out", str(out),
                         "--npoints", "8", "--mesh", "8",
                         "--db", str(dbp)]) == 0
        db = ResultsDB(str(dbp))
        rows = db.history(material="mos2_kp")
        assert len(rows) == 1
        assert rows[0]["gap_eV"] == pytest.approx(1.67, abs=1e-6)

    def test_history_command(self, tmp_path, capsys):
        dbp = tmp_path / "res.db"
        cli.main(["run", "mos2_kp", "--out", str(tmp_path / "r"),
                  "--npoints", "8", "--mesh", "8", "--db", str(dbp)])
        assert cli.main(["history", "--db", str(dbp)]) == 0
        out = capsys.readouterr().out
        assert "mos2_kp" in out and "1.670" in out
