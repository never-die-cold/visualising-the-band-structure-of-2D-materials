"""Allocation bounds and streamed input contract; no timing thresholds."""
from pathlib import Path
import tracemalloc

import numpy as np
import pytest

from core.parser import VASPEigenvalParser
from core.dos_analyzer import DosAnalyzer
from examples.benchmark_bandviz import write_fixture
from vdw_studio.engine.models import HoneycombModel
from vdw_studio.engine.solver import solve_dos


def test_standard_parser_streams_without_loading_full_text(tmp_path, monkeypatch):
    path = tmp_path / 'EIGENVAL'
    write_fixture(path, 1024, 32)

    def forbidden(*args, **kwargs):
        raise AssertionError('Full-text read is forbidden for EIGENVAL')
    monkeypatch.setattr(Path, 'read_text', forbidden)
    tracemalloc.start()
    try:
        data = VASPEigenvalParser(str(path)).parse()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    expected = ((np.arange(32) - 16) * .15)[None, :] + .02 * np.cos(2*np.pi*np.arange(1024)/1024)[:, None]
    np.testing.assert_allclose(data.energies, expected, atol=6e-13, rtol=0)
    output_bytes = sum(a.nbytes for a in (data.energies, data.occupations, data.kpoints, data.weights, data.kdistances))
    assert peak < 2*output_bytes + 256*1024


def test_gaussian_workspace_is_reused_within_budget():
    energies = np.tile(np.linspace(-2, 2, 32), (1024, 1))
    tracemalloc.start()
    try:
        spectrum = DosAnalyzer(energies, max_workspace_mb=.5).calculate_dos((-4, 4), 401, .1)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Workspace + shifted states, weights, occupied mask + small grid/allocator overhead.
    assert peak < .5 * 1024**2 + 3*energies.nbytes + 256*1024
    assert spectrum.integral == pytest.approx(32, abs=1e-9)
    np.testing.assert_allclose(spectrum.total_dos, spectrum.vb_dos+spectrum.cb_dos, atol=1e-12)


def test_vdw_dos_workspace_size_preserves_gaussian_result():
    model = HoneycombModel(2.46)
    small = solve_dos(model, mesh=(8, 8), n_points=401, max_workspace_mb=.004)
    large = solve_dos(model, mesh=(8, 8), n_points=401, max_workspace_mb=32.)
    np.testing.assert_allclose(small.dos, large.dos, atol=1e-13)
    with pytest.raises(ValueError, match='workspace'):
        solve_dos(model, max_workspace_mb=0)
