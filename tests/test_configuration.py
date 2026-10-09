import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from storage.config_manager import ConfigManager, DEFAULT_CONFIG
from storage.database import Database
from vdw_studio.storage import ResultsDB
from utils.paths import app_data_dir


def test_config_instances_defaults_and_returned_values_are_independent(tmp_path):
    first, second = ConfigManager(tmp_path/'a.json'), ConfigManager(tmp_path/'b.json')
    first.set('energy_range.min', -3.)
    first.add_recent_file(str(tmp_path/'file'))
    returned = first.get_all()
    returned['energy_range']['min'] = -99.
    first.get('recent_files').clear()
    assert first.get_energy_range() == (-3., 5.)
    assert first.get_recent_files()
    assert second.get_all() == DEFAULT_CONFIG
    assert DEFAULT_CONFIG['energy_range']['min'] == -5. and not DEFAULT_CONFIG['recent_files']


def test_partial_nested_configuration_and_invalid_fields_are_repaired(tmp_path):
    path = tmp_path/'settings.json'
    path.write_text(json.dumps({'energy_range': {'min': -3.}, 'window_geometry': {'width': 800},
        'dos_sigma': float('nan'), 'recent_files': 'wrong', 'dos_show_total': 1,
        'extra': {'mode': 'custom'}}), encoding='utf-8')
    config = ConfigManager(path)
    assert config.get_energy_range() == (-3., 5.)
    assert config.get('window_geometry.width') == 800 and config.get('window_geometry.height') == 900
    assert config.get('dos_sigma') == .05 and config.get('recent_files') == []
    assert config.get('dos_show_total') is True and config.get('extra.mode') == 'custom'
    assert len(config.diagnostics) == 3


@pytest.mark.parametrize('text', ['[]', '{broken', '{"energy_range":{"min":5,"max":-5}}'])
def test_corrupt_or_inverted_configuration_recovers_without_overwriting_source(tmp_path, text):
    path = tmp_path/'settings.json'
    path.write_text(text, encoding='utf-8')
    config = ConfigManager(path)
    assert config.get_energy_range() == (-5., 5.)
    assert path.read_text(encoding='utf-8') == text
    assert config.diagnostics


@pytest.mark.parametrize('updates', [{'dos_sigma': 0}, {'dos_sigma': float('nan')},
    {'energy_range': {'min': 5., 'max': 2.}}, {'window_geometry': {'width': -1}},
    {'recent_files': [None]}])
def test_invalid_update_preserves_memory_and_file(tmp_path, updates):
    path = tmp_path/'settings.json'
    config = ConfigManager(path)
    original = path.read_bytes()
    before = config.get_all()
    with pytest.raises(ValueError):
        config.update(updates)
    assert path.read_bytes() == original and config.get_all() == before


def test_atomic_replace_failure_keeps_previous_file_and_cleans_temporary(tmp_path, monkeypatch):
    from utils import atomic_io
    path = tmp_path/'settings.json'
    config = ConfigManager(path)
    before = path.read_bytes()
    def interrupted(*args):
        raise PermissionError('simulated interrupted replacement')
    monkeypatch.setattr(atomic_io.os, 'replace', interrupted)
    with pytest.raises(PermissionError):
        config.set('dos_sigma', .2)
    assert path.read_bytes() == before and config.get('dos_sigma') == .05
    assert sorted(p.name for p in tmp_path.iterdir()) == ['settings.json']


def test_energy_window_update_is_one_atomic_change_and_nested_merge_keeps_other_values(tmp_path):
    config = ConfigManager(tmp_path/'settings.json')
    config.set_energy_range(10., 20.)
    config.update({'window_geometry': {'width': 800}})
    assert config.get_energy_range() == (10., 20.)
    assert config.get('window_geometry.height') == 900
    assert ConfigManager(config.config_path).get_all() == config.get_all()


def test_default_state_migrates_legacy_once_and_never_writes_user_source(tmp_path, monkeypatch):
    workspace, state = tmp_path/'workspace', tmp_path/'state'
    workspace.mkdir()
    (workspace/'config').mkdir()
    config_source = workspace/'config'/'settings.json'
    config_source.write_text('{"dos_sigma":0.2}', encoding='utf-8')
    database_source = workspace/'data'/'bandviz.db'
    original = Database(database_source)
    original.add_task('legacy', '/input/EIGENVAL', band_gap=1.1)
    before = hashlib.sha256(database_source.read_bytes()).hexdigest()
    monkeypatch.chdir(workspace)
    monkeypatch.setenv('BANDVIZ_DATA_DIR', str(state))
    config, database = ConfigManager(), Database()
    assert config.config_path == state/'settings.json' and config.get('dos_sigma') == .2
    assert database.db_path == state/'bandviz.db' and database.list_tasks()[0]['name'] == 'legacy'
    assert config_source.read_text() == '{"dos_sigma":0.2}'
    assert hashlib.sha256(database_source.read_bytes()).hexdigest() == before
    config.set('dos_sigma', .3)
    config_source.write_text('{"dos_sigma":0.4}')
    assert ConfigManager().get('dos_sigma') == .3
    assert app_data_dir() == state


@pytest.mark.parametrize('kind', ['bandviz', 'vdw'])
def test_every_database_operation_closes_connection(tmp_path, monkeypatch, kind):
    connections = []
    original = sqlite3.connect
    class Tracked(sqlite3.Connection):
        closed = False
        def close(self):
            self.closed = True
            return super().close()
    def connect(*args, **kwargs):
        result = original(*args, factory=Tracked, **kwargs)
        connections.append(result)
        return result
    monkeypatch.setattr(sqlite3, 'connect', connect)
    if kind == 'bandviz':
        db = Database(tmp_path/'tasks.db')
        for i in range(20):
            db.add_task(str(i), str(i))
            db.list_tasks()
    else:
        db = ResultsDB(tmp_path/'runs.db')
        for i in range(20):
            db.record(str(i), 'tb')
            db.history()
    assert len(connections) == 41 and all(c.closed for c in connections)
    with pytest.raises(sqlite3.ProgrammingError, match='closed'):
        connections[-1].execute('SELECT 1')


@pytest.mark.parametrize('factory', [Database, ResultsDB])
def test_future_database_schema_is_refused_without_downgrading(tmp_path, factory):
    path = tmp_path/'future.db'
    with sqlite3.connect(path) as conn:
        conn.execute('PRAGMA user_version=99')
    before = path.read_bytes()
    with pytest.raises(ValueError, match='schema version'):
        factory(path)
    assert path.read_bytes() == before


def test_gui_settings_write_failure_keeps_event_loop_and_close_functional(tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QApplication
    from test_lifecycle import root_window
    from utils import atomic_io
    app = QApplication.instance() or QApplication([])
    window = root_window(tmp_path, monkeypatch)
    previous = window.config.config_path.read_bytes()
    def failure(*args):
        raise PermissionError('simulated disk failure')
    monkeypatch.setattr(atomic_io.os, 'replace', failure)
    window.show()
    window.set_fermi_level(.2)
    app.processEvents()
    assert 'previous file retained' in window.statusBar().currentMessage()
    assert window.config.config_path.read_bytes() == previous
    window.close()
    app.processEvents()
    assert not window.isVisible()
