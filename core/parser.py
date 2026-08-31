import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class BandData:
    """能带数据结构"""
    kpoints: np.ndarray                    # (nk, 3) k 点坐标
    kdistances: np.ndarray                 # (nk,) 累积 k 点距离
    energies: np.ndarray                   # (nk, nbands) 能量
    num_bands: int
    num_electrons: int
    kpoint_labels: List[Tuple[int, str]] = field(default_factory=list)  # (index, label)


# 高对称点标准标签映射
H_SYMBOLS = {
    'GAMMA': 'Γ',
    'G': 'Γ',
    'GAM': 'Γ',
    'K': 'K',
    'M': 'M',
    'L': 'L',
    'X': 'X',
    'W': 'W',
    'A': 'A',
    'H': 'H',
    'R': 'R',
    'S': 'S',
    'T': 'T',
    'U': 'U',
    'Y': 'Y',
    'Z': 'Z',
    'D': 'D',
    'SIGMA': 'Σ',
    'DELTA': 'Δ',
    'LAMBDA': 'Λ',
}


def _format_label(raw_label: str) -> Optional[str]:
    """格式化高对称点标签"""
    if not raw_label:
        return None
    label = raw_label.strip().upper()
    # 去掉引号
    label = label.strip('"\'')
    if label in H_SYMBOLS:
        return H_SYMBOLS[label]
    # 如果已经包含希腊字母等，直接返回
    return raw_label.strip()


def _extract_label_from_line(line: str) -> Optional[str]:
    """从 KPOINTS 行中提取标签（支持 ! 和 # 注释）"""
    # 优先取注释部分
    for marker in ['!', '#']:
        if marker in line:
            comment = line.split(marker, 1)[1].strip()
            label = _format_label(comment)
            return label
    return None


class VASPKpointsParser:
    """解析 VASP KPOINTS 文件，提取高对称点标签索引"""

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)

    def parse(self) -> List[Tuple[int, str]]:
        if not self.filepath.exists():
            return []

        with open(self.filepath, 'r') as f:
            raw_lines = f.readlines()

        # 保留空行（用于 line-mode 分段），仅去掉纯注释行
        lines = [line.rstrip('\n').rstrip('\r') for line in raw_lines]
        content = [line.strip() for line in lines if not line.strip().startswith('#')]

        if len(content) < 4:
            return []

        # 判断模式
        mode_line = content[2].lower() if len(content) > 2 else ''
        coord_line = ''
        for i in range(1, min(len(content), 6)):
            if content[i].lower() in ('reciprocal', 'cartesian', 'rec', 'car'):
                mode_line = content[i - 1].lower() if i > 0 else ''
                coord_line = content[i].lower()
                break

        if 'line' in mode_line:
            return self._parse_line_mode(content)
        else:
            return self._parse_explicit_mode(content)

    def _find_data_start(self, content: List[str]) -> int:
        """找到 k-point 数据起始行索引"""
        for i, line in enumerate(content):
            stripped = line.lower()
            if stripped in ('reciprocal', 'cartesian', 'rec', 'car'):
                return i + 1
        return 3  # fallback

    def _parse_line_mode(self, content: List[str]) -> List[Tuple[int, str]]:
        """解析 Line-Mode KPOINTS"""
        try:
            num_per_segment = int(content[1].split()[0])
        except ValueError:
            return []

        data_start = self._find_data_start(content)
        if data_start <= 0 or data_start >= len(content):
            return []

        # 按空行分组，每组为一对端点
        segment_pairs = []
        current_buf = []

        for i in range(data_start, len(content)):
            stripped = content[i]
            if not stripped:
                if len(current_buf) == 2:
                    segment_pairs.append((current_buf[0], current_buf[1]))
                current_buf = []
                continue
            if stripped.startswith('#'):
                continue
            current_buf.append(stripped)

        if len(current_buf) == 2:
            segment_pairs.append((current_buf[0], current_buf[1]))

        if not segment_pairs:
            return []

        labels = []
        segment_length = num_per_segment - 1

        for seg_idx, (line1, line2) in enumerate(segment_pairs):
            label1 = _extract_label_from_line(line1)
            label2 = _extract_label_from_line(line2)
            start_idx = seg_idx * segment_length
            end_idx = (seg_idx + 1) * segment_length

            if label1 and start_idx not in [idx for idx, _ in labels]:
                labels.append((start_idx, label1))
            if label2 and end_idx not in [idx for idx, _ in labels]:
                labels.append((end_idx, label2))

        return labels

    def _parse_explicit_mode(self, content: List[str]) -> List[Tuple[int, str]]:
        """解析 Explicit/Automatic KPOINTS，按行索引给标签"""
        try:
            num_kpoints = int(content[1].split()[0])
        except ValueError:
            return []

        data_start = self._find_data_start(content)

        labels = []
        for i in range(num_kpoints):
            line_idx = data_start + i
            if line_idx >= len(content):
                break
            line = content[line_idx]
            if not line or line.startswith('#'):
                continue
            label = _extract_label_from_line(line)
            if label:
                labels.append((i, label))

        return labels


class VASPEigenvalParser:
    """解析 VASP EIGENVAL 文件"""

    def __init__(self, filepath: str, kpoints_path: Optional[str] = None):
        self.filepath = Path(filepath)
        self.kpoints_path = Path(kpoints_path) if kpoints_path else None

    def parse(self) -> BandData:
        with open(self.filepath, 'r') as f:
            lines = f.readlines()

        if len(lines) < 6:
            raise ValueError(f"EIGENVAL file too short: {self.filepath}")

        # 解析第 6 行头部（1-based 第 6 行，0-based 第 5 行）
        header_line = lines[5].strip()
        header_parts = header_line.split()

        if len(header_parts) >= 4:
            # 标准格式: natoms natoms nbands nelectrons
            _, num_atoms, num_bands, num_electrons = map(int, header_parts[:4])
        elif len(header_parts) == 3:
            # 兼容旧格式: natoms nbands nelectrons
            num_atoms, num_bands, num_electrons = map(int, header_parts[:3])
        else:
            raise ValueError(f"Cannot parse EIGENVAL header line 6: {header_line}")

        kpoints = []
        energies = []
        line_idx = 6

        while line_idx < len(lines):
            line = lines[line_idx].strip()
            if not line:
                line_idx += 1
                continue

            # k 点信息行: kx ky kz weight
            k_info = line.split()
            if len(k_info) < 4:
                raise ValueError(
                    f"Invalid k-point line at {line_idx + 1} in {self.filepath}"
                )

            kx, ky, kz = map(float, k_info[:3])
            kpoints.append([kx, ky, kz])

            line_idx += 1
            band_energies = []

            # 读取该 k 点的所有能带
            for _ in range(num_bands):
                if line_idx >= len(lines):
                    break

                parts = lines[line_idx].strip().split()
                if len(parts) >= 2:
                    # 第二列为能量（第一列为能带序号）
                    band_energies.append(float(parts[1]))

                line_idx += 1

            if len(band_energies) != num_bands:
                raise ValueError(
                    f"Expected {num_bands} band energies at k-point "
                    f"{len(kpoints)}, got {len(band_energies)}"
                )

            energies.append(band_energies)

        kpoints = np.array(kpoints)
        energies = np.array(energies)

        if len(kpoints) == 0 or len(energies) == 0:
            raise ValueError(f"No valid k-points or energies found in {self.filepath}")

        # 计算 k 点累积距离
        kdistances = self._calc_kdistances(kpoints)

        # 解析 KPOINTS 标签
        kpoint_labels = []
        if self.kpoints_path and self.kpoints_path.exists():
            kpoint_labels = VASPKpointsParser(str(self.kpoints_path)).parse()
        else:
            # 尝试在同一目录下查找 KPOINTS
            auto_kpoints = self.filepath.parent / "KPOINTS"
            if auto_kpoints.exists():
                kpoint_labels = VASPKpointsParser(str(auto_kpoints)).parse()

        return BandData(
            kpoints=kpoints,
            kdistances=kdistances,
            energies=energies,
            num_bands=num_bands,
            num_electrons=num_electrons,
            kpoint_labels=kpoint_labels
        )

    def _calc_kdistances(self, kpoints: np.ndarray) -> np.ndarray:
        """计算 k 点路径累积距离"""
        distances = [0.0]
        for i in range(1, len(kpoints)):
            dk = np.linalg.norm(kpoints[i] - kpoints[i - 1])
            distances.append(distances[-1] + dk)
        return np.array(distances)
