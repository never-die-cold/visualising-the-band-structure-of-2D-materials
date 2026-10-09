"""Immutable task metadata, with isolated runtime models and aligned structures."""
from dataclasses import asdict, dataclass, is_dataclass
from copy import deepcopy
import json
from uuid import uuid4

import numpy as np

from .engine.kpath import KPath
from .structure.crystal import Atom, Crystal
from .structure.lattice import Lattice


def align_structure(base, model):
    """Use the model's actual lattice, retaining fractional geometry for TB.

    SK models also supply explicit atom sites; local k·p keeps the supplied
    material structure. Rebuild from base each time, so strain never accumulates.
    """
    result = deepcopy(base)
    if hasattr(model, 'lattice'):
        result.lattice = Lattice(model.lattice.matrix.copy())
    if hasattr(model, 'sites'):
        result.atoms = [Atom(symbol, np.asarray(frac).copy()) for symbol, frac in model.sites]
    return result


def _json_state(value):
    if isinstance(value, np.ndarray):
        return _json_state(value.tolist())
    if isinstance(value, np.generic):
        return _json_state(value.item())
    if isinstance(value, complex):
        return {'real': value.real, 'imag': value.imag}
    if is_dataclass(value):
        return _json_state(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_state(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_state(v) for v in value]
    if hasattr(value, '__dict__'):
        return {'class': type(value).__module__ + '.' + type(value).__name__,
                'state': _json_state(vars(value))}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f'Unsupported model state: {type(value).__name__}')


def encode_model_state(model):
    return json.dumps(_json_state(model), sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class SimulationSnapshot:
    task_id: str
    material_key: str
    material_name: str
    engine: str
    source: str
    n_valence: int
    lattice_rows: tuple
    atoms: tuple
    strain: tuple
    electric_field: float
    n_per_segment: int
    dos_mesh: tuple
    dos_sigma: float
    gap_mesh: tuple
    path: tuple
    model_state_json: str
    gap_reference: tuple | None
    gap_note: str
    include_valley: bool

    @classmethod
    def capture(cls, preset, model, structure, *, strain=(0., 0.), electric_field=0.,
                n_per_segment=40, dos_mesh=(48, 48), dos_sigma=.05,
                gap_mesh=(24, 24), include_valley=False):
        structure = align_structure(structure, model)
        lattice = structure.lattice
        kpath = KPath.for_lattice(lattice)
        if preset.engine == 'kp':
            from .simulation import KP_QMAX
            basis = lattice.reciprocal_matrix[:2, :2]
            k = np.array([1/3, 1/3]) @ basis
            m = np.array([.5, 0]) @ basis
            endpoints = np.array([k - KP_QMAX * k / np.linalg.norm(k), k,
                                  k + KP_QMAX * (m-k) / np.linalg.norm(m-k)]) @ np.linalg.inv(basis)
            labels = ('→Γ', 'K', '→M')
            path = (labels, tuple((name, tuple(map(float, point))) for name, point in zip(labels, endpoints)))
        else:
            path = (tuple(kpath.path), tuple((name, tuple(map(float, point))) for name, point in kpath.points.items()))
        return cls(str(uuid4()), preset.key, preset.name, preset.engine, preset.source,
                   preset.n_valence, tuple(tuple(map(float, row)) for row in lattice.matrix),
                   tuple((a.symbol, tuple(map(float, a.frac)), a.label) for a in structure.atoms),
                   tuple(map(float, strain)), float(electric_field), int(n_per_segment),
                   tuple(dos_mesh), float(dos_sigma), tuple(gap_mesh), path,
                   encode_model_state(model),
                   None if preset.gap_ref is None else tuple(preset.gap_ref),
                   preset.gap_note, bool(include_valley))

    def make_structure(self):
        return Crystal(Lattice(self.lattice_rows),
                       [Atom(symbol, np.array(frac), label) for symbol, frac, label in self.atoms])

    def to_dict(self):
        result = asdict(self)
        result['model_state'] = json.loads(result.pop('model_state_json'))
        return result
