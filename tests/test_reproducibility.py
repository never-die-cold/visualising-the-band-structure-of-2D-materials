import csv
import json
import sqlite3
from dataclasses import replace

import numpy as np
import pytest

from vdw_studio import cli
from vdw_studio.engine.models import HoneycombModel, TightBindingModel, Hopping
from vdw_studio.engine.strain import apply_strain
from vdw_studio.presets import get_preset
from vdw_studio.reproducibility import provenance, task_key, restore_model
from vdw_studio.storage import ResultsDB
from vdw_studio.task_state import SimulationSnapshot
from vdw_studio.structure.lattice import Lattice


def snapshot(key='graphene'):
    preset = get_preset(key)
    model = preset.make_model()
    if key == 'graphene':
        model = apply_strain(model, ex=.02, ey=-.01)
    task = SimulationSnapshot.capture(preset, model, preset.make_structure(preset.structure_key),
        n_per_segment=4, dos_mesh=(4, 4), gap_mesh=(4, 4)).to_dict()
    return task, model


@pytest.mark.parametrize('key', ['graphene', 'phosphorene', 'mos2_kp'])
def test_stored_numerical_model_restores_without_current_preset_factory(key, monkeypatch):
    task, model = snapshot(key)
    from vdw_studio.presets import PRESETS
    def forbidden():
        raise AssertionError('Replay must use saved parameters')
    monkeypatch.setitem(PRESETS, key, replace(get_preset(key), make_model=forbidden))
    restored = restore_model(json.loads(json.dumps(task)))
    if key.endswith('kp'):
        for q in ([0, 0], [.071, -.037]):
            np.testing.assert_array_equal(restored.hamiltonian(q), model.hamiltonian(q))
    else:
        for k in ([0, 0], [.173, .287], [.719, .613]):
            np.testing.assert_array_equal(restored.hamiltonian(k), model.hamiltonian(k))
        np.testing.assert_array_equal(restored.lattice.matrix, model.lattice.matrix)


def test_task_identity_ignores_uuid_but_includes_parameters_and_software():
    task, _ = snapshot()
    code = provenance()
    first = task_key(task, code)
    task['task_id'] = 'different-invocation'
    assert task_key(task, code) == first
    task['dos_sigma'] *= 2
    assert task_key(task, code) != first
    task['dos_sigma'] /= 2
    code['dependencies']['numpy'] = 'different'
    assert task_key(task, code) != first


def test_record_cannot_request_dynamic_import():
    task, _ = snapshot()
    task['model_state']['class'] = 'os.system'
    with pytest.raises(ValueError, match='not supported'):
        restore_model(task)


def test_old_database_migrates_preserving_indirect_result(tmp_path):
    path = tmp_path / 'old.db'
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE runs (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, material TEXT NOT NULL, engine TEXT NOT NULL, formula TEXT, gap_eV REAL, gap_direct INTEGER, payload TEXT)')
        conn.execute("INSERT INTO runs(timestamp,material,engine,gap_eV,gap_direct,payload) VALUES('old','indirect','tb',.9,0,'{}')")
    db = ResultsDB(path)
    row = db.history()[0]
    assert row['gap_direct'] is False and row['status'] == 'completed'
    first = db.record('retry', 'tb', task_key='same', status='failed', error='interrupted')
    second = db.record('retry', 'tb', task_key='same', gap_eV=.9, gap_direct=False)
    assert second == first and len(db.history()) == 2
    assert db.find_task('same')['status'] == 'completed'
    output = tmp_path / 'records.csv'
    assert db.export_csv(output) == 2
    with output.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]['gap_direct'] == 'False'
    assert json.loads(rows[0]['payload']) == {}


def test_cli_resume_requires_intact_artifacts_and_same_task(tmp_path, monkeypatch):
    out, db = tmp_path / 'run', tmp_path / 'runs.db'
    args = ['run', 'mos2_kp', '--out', str(out), '--db', str(db), '--npoints', '4', '--mesh', '4']
    assert cli.main(args) == 0
    summary = json.loads((out/'summary.json').read_text(encoding='utf-8'))
    assert summary['provenance']['source_sha256'] and summary['task_key']
    original = cli.simulate
    calls = []
    def tracked(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(cli, 'simulate', tracked)
    assert cli.main(args + ['--resume']) == 0 and not calls
    assert len(ResultsDB(db).history()) == 1
    (out/'bands.npz').write_bytes(b'corrupt output')
    assert cli.main(args + ['--resume']) == 0 and len(calls) == 1
    assert cli.main(args + ['--sigma', '.1', '--resume']) == 0 and len(calls) == 2


def test_failed_cli_task_is_recorded_and_retried(tmp_path, monkeypatch):
    args = ['run', 'mos2_kp', '--out', str(tmp_path/'out'), '--db', str(tmp_path/'runs.db'),
            '--npoints', '4', '--mesh', '4']
    original = cli.simulate
    def fail(*args, **kwargs):
        raise ValueError('numerical failure')
    monkeypatch.setattr(cli, 'simulate', fail)
    with pytest.raises(ValueError, match='numerical failure'):
        cli.main(args)
    row = ResultsDB(tmp_path/'runs.db').history()[0]
    assert row['status'] == 'failed' and 'numerical failure' in row['error']
    monkeypatch.setattr(cli, 'simulate', original)
    assert cli.main(args+['--resume']) == 0
    recovered = ResultsDB(tmp_path/'runs.db').history()
    assert len(recovered) == 1 and recovered[0]['id'] == row['id']
    assert recovered[0]['status'] == 'completed'


def test_cli_replay_rebuilds_saved_kp_bands(tmp_path):
    out = tmp_path/'original'
    assert cli.main(['run', 'mos2_kp', '--out', str(out), '--npoints', '4', '--mesh', '4']) == 0
    replay = tmp_path/'replay'
    assert cli.main(['replay', str(out/'summary.json'), '--out', str(replay)]) == 0
    with np.load(out/'bands.npz') as first, np.load(replay/'bands.npz') as second:
        np.testing.assert_array_equal(first['energies'], second['energies'])
    report = json.loads((replay/'replay.json').read_text(encoding='utf-8'))
    assert report['same_source'] is True and report['gap']['gap_eV'] == report['original_gap']['gap_eV']


def test_screening_records_actual_indirect_gap_and_can_resume(tmp_path, monkeypatch):
    from examples import run_screening as screen
    # Diagonal periodic TB: VBM at (0,0), CBM at (1/2,0), exact gap 0.9.
    model = TightBindingModel(Lattice.square(2), 2, ['A','B'], [-1.2, 1.7], [
        Hopping(0,0,(1,0,0),.5), Hopping(0,0,(-1,0,0),.5),
        Hopping(1,1,(1,0,0),.5), Hopping(1,1,(-1,0,0),.5)])
    monkeypatch.setattr(screen, 'TB_MATERIALS', {'hbn': (lambda: model, 1)})
    monkeypatch.setattr(screen, 'STRAINS', (0.,))
    db = ResultsDB(tmp_path/'screen.db')
    assert screen.screen_bands(db, mesh=(4,4)) == 0
    row = db.history()[0]
    assert row['gap_eV'] == pytest.approx(.9, abs=1e-8) and row['gap_direct'] is False
    assert row['payload']['gap']['search_metadata']['converged'] is True
    assert screen.screen_bands(db, mesh=(4,4), resume=True) == 0
    assert len(db.history()) == 1
