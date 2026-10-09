import warnings

import matplotlib
from matplotlib import pyplot as plt

from vdw_studio.visualization import fonts, plot_structure, plot_band_structure, plot_dos, plot_bz_path
from vdw_studio.engine.models import HoneycombModel
from vdw_studio.engine.kpath import KPath
from vdw_studio.engine.solver import solve_bands, solve_dos
from vdw_studio.structure.builders import graphene


def test_all_export_plot_titles_render_without_cjk_fonts(tmp_path, monkeypatch):
    monkeypatch.setattr(fonts, 'setup_cjk_fonts', lambda: False)
    model = HoneycombModel(2.46)
    path = KPath.for_lattice(model.lattice)
    bands = solve_bands(model, path, 4)
    dos = solve_dos(model, mesh=(4,4), n_points=101)
    with matplotlib.rc_context({'font.family': 'DejaVu Sans', 'font.sans-serif': ['DejaVu Sans']}):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always')
            plots = [plot_structure(graphene(), title='石墨烯结构'),
                     plot_band_structure(bands, title='石墨烯能带'),
                     plot_dos(dos, title='石墨烯态密度'),
                     plot_bz_path(model.lattice, path, title='石墨烯布里渊区')]
            try:
                assert all(not any('\u3400' <= ch <= '\u9fff' for ch in ax.get_title()) for _, ax in plots)
                for i, (fig, _) in enumerate(plots):
                    fig.savefig(tmp_path/f'{i}.png')
                assert not any('Glyph' in str(item.message) for item in recorded)
            finally:
                for fig, _ in plots:
                    plt.close(fig)


def test_found_cjk_font_preserves_user_title(monkeypatch):
    monkeypatch.setattr(fonts, 'setup_cjk_fonts', lambda: True)
    assert fonts.plot_text('石墨烯 — bands', 'graphene — bands') == '石墨烯 — bands'
