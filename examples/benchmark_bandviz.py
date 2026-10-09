"""Synthetic scaling benchmark; never imports the paper generation script."""
import argparse
import io
import json
import platform
from pathlib import Path
import time
import tracemalloc

import numpy as np

from core.parser import VASPEigenvalParser
from core.dos_analyzer import DosAnalyzer


def measured(function):
    tracemalloc.start()
    start = time.perf_counter()
    result = function()
    seconds = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, {'seconds': seconds, 'peak_traced_MiB': peak / 1024**2}


def write_fixture(path, nk, nb):
    with path.open('w', encoding='utf-8') as stream:
        stream.write(f'1 1 1 1\n1 1 1 1 0\n0\nCAR\nSynthetic scaling benchmark\n{nb} {nk} {nb}\n')
        for k in range(nk):
            stream.write(f'\n{k / nk:.12f} 0 0 {1 / nk:.12f}\n')
            for band in range(nb):
                energy = (band - nb / 2) * .15 + .02 * np.cos(2 * np.pi * k / nk)
                stream.write(f'{band+1} {energy:.12f} {2 if band < nb//2 else 0}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='results/benchmark')
    args = parser.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    report = {'python': platform.python_version(), 'platform': platform.platform(),
              'numpy': np.__version__, 'memory_metric': 'tracemalloc peak, includes NumPy arrays; excludes Qt/OS/native allocations',
              'baseline': 'Same validated parser with the entire decoded text resident in StringIO; not the historical parser',
              'energy_points': 401, 'dos_workspace_MiB': 4., 'cases': []}
    for name, nk, nb in [('small', 64, 8), ('medium', 1024, 32), ('large', 4096, 64)]:
        folder = root / name
        folder.mkdir(exist_ok=True)
        path = folder / 'EIGENVAL'
        write_fixture(path, nk, nb)
        reader = VASPEigenvalParser(str(path))
        data, streaming = measured(reader.parse)

        def full_text():
            text = path.read_text(encoding='utf-8-sig')
            values = reader._read_stream(io.StringIO(text))
            return reader._assemble_data(*values)
        reference, baseline = measured(full_text)
        np.testing.assert_array_equal(data.energies, reference.energies)
        expected = ((np.arange(nb) - nb/2) * .15)[None, :] + .02 * np.cos(2*np.pi*np.arange(nk)/nk)[:, None]
        np.testing.assert_allclose(data.energies, expected, atol=6e-13, rtol=0)
        spectrum, dos = measured(lambda: DosAnalyzer.from_band_data(
            data, max_workspace_mb=4.).calculate_dos((-7, 7), 401, .1))
        assert abs(spectrum.integral - 2*nb) < 1e-6
        np.testing.assert_allclose(spectrum.total_dos, spectrum.vb_dos+spectrum.cb_dos, atol=1e-12)
        case = {'name': name, 'nk': nk, 'bands': nb, 'file_MiB': path.stat().st_size / 1024**2,
                'output_arrays_MiB': sum(a.nbytes for a in (data.kpoints, data.kdistances, data.weights,
                    data.energies, data.occupations))/1024**2,
                'streaming': streaming, 'full_text_reference': baseline, 'dos': dos,
                'dos_integral': spectrum.integral}
        report['cases'].append(case)
        print(json.dumps(case), flush=True)
    (root / 'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
