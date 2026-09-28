"""示例：全部内置材料 — 结构建模 + POSCAR/XYZ 导出。

运行::

    python examples/export_all_structures.py

输出保存到 ``examples/output/structures/``：每个材料一个 POSCAR
（可直接作为 VASP 输入）与一个 XYZ。
"""

from pathlib import Path

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from vdw_studio.io import write_poscar, write_xyz
from vdw_studio.presets import get_preset, list_presets


def main() -> None:
    out = Path(__file__).parent / "output" / "structures"
    out.mkdir(parents=True, exist_ok=True)

    print(f"{'材料':<16s}{'化学式':<10s}{'原子数':<8s}{'晶格常数 (Å)'}")
    print("-" * 60)
    for key in list_presets():
        p = get_preset(key)
        c = p.make_structure(p.structure_key)
        write_poscar(c, str(out / f"{p.structure_key}.POSCAR"),
                     title=f"{p.name} (vdW Studio)")
        write_xyz(c, str(out / f"{p.structure_key}.xyz"), title=p.name)
        a, b, cc, _, _, gamma = c.lattice.parameters()
        print(f"{p.name:<16s}{c.formula_str:<10s}{c.n_atoms:<8d}"
              f"a={a:.4f}, b={b:.4f}, γ={gamma:.2f}°")
    print(f"\n已导出 POSCAR + XYZ 到 {out}")


if __name__ == "__main__":
    main()
