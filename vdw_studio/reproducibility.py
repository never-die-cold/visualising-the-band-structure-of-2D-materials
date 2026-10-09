"""Versioned provenance and safe reconstruction of supported stored models."""
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

SCHEMA_VERSION = 1


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode('utf-8')).hexdigest()


def provenance():
    root = Path(__file__).resolve().parent
    files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
             for p in sorted(root.rglob('*.py'))}
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
            stderr=subprocess.DEVNULL, text=True, timeout=3).strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    versions = {}
    for name in ('band-structure-2d', 'numpy', 'scipy', 'matplotlib', 'PyQt5'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'schema_version': SCHEMA_VERSION, 'git_revision': revision,
            'source_sha256': canonical_hash(files), 'source_files': files,
            'source_hash_encoding': 'UTF-8 source bytes with LF line endings',
            'python': platform.python_version(), 'dependencies': versions}


def task_key(task, code):
    inputs = deepcopy(task)
    inputs.pop('task_id', None)
    return canonical_hash({'task': inputs, 'source_sha256': code['source_sha256'],
                           'schema_version': SCHEMA_VERSION, 'python': code.get('python'),
                           'dependencies': code.get('dependencies')})


def restore_model(task):
    """Restore numerical state without importing classes supplied by the record."""
    from .engine.models import TightBindingModel, Hopping
    from .engine.kp_tmd import TMDKpModel, TMDKpParams
    from .structure.lattice import Lattice
    encoded = task['model_state']
    state = encoded['state']
    if task['engine'] == 'tb' and encoded['class'] in {
        'vdw_studio.engine.models.' + name for name in (
            'TightBindingModel', 'HoneycombModel', 'BuckledHoneycombModel',
            'BilayerGrapheneModel', 'PhosphoreneRudenko')}:
        return TightBindingModel(Lattice(state['lattice']['state']['matrix']),
            state['n_sites'], state['site_symbols'], state['onsite'],
            [Hopping(**hop) for hop in state['hoppings']], name=state['name'])
    if task['engine'] == 'kp' and encoded['class'] == 'vdw_studio.engine.kp_tmd.TMDKpModel':
        return TMDKpModel(TMDKpParams(**state['params']))
    raise ValueError('Stored model is not supported for validated replay')


def replay_task(task):
    from .simulation import simulate
    from .structure.lattice import Lattice
    return simulate(restore_model(task), n_per_segment=task['n_per_segment'],
        dos_mesh=tuple(task['dos_mesh']), dos_sigma=task['dos_sigma'],
        n_valence=task['n_valence'], lattice=Lattice(task['lattice_rows']),
        gap_mesh=tuple(task['gap_mesh']), include_valley=task['include_valley'])
