"""Strict readers for VASP EIGENVAL and explicit/line-mode KPOINTS."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import warnings

import numpy as np


@dataclass
class BandData:
    """All spin bands are stored as 2D columns: up bands, then down bands.

    num_bands counts columns; nbands is VASP's NBANDS per spin. The spin views
    have shape (nk, NBANDS, ISPIN). With a lattice, distances are in Å⁻¹;
    otherwise they are explicitly uncalibrated fractional-coordinate distances.
    """

    kpoints: np.ndarray
    kdistances: np.ndarray
    energies: np.ndarray
    num_bands: int
    num_electrons: float
    kpoint_labels: List[Tuple[int, str]] = field(default_factory=list)
    ispin: int = 1
    occupations: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    segments: List[Tuple[int, int]] = field(default_factory=list)  # inclusive ends
    input_format: str = "vasp"
    lattice_matrix: Optional[np.ndarray] = None  # row vectors, Å
    lattice_source: Optional[str] = None
    sampling_kind: str = "unknown"  # path, mesh, explicit, unknown

    @property
    def reciprocal_matrix(self) -> Optional[np.ndarray]:
        if self.lattice_matrix is None:
            return None
        return 2 * np.pi * np.linalg.inv(self.lattice_matrix).T

    @property
    def kcartesian(self) -> Optional[np.ndarray]:
        reciprocal = self.reciprocal_matrix
        return None if reciprocal is None else self.kpoints @ reciprocal

    @property
    def distance_unit(self) -> str:
        return "angstrom^-1" if self.lattice_matrix is not None else "fractional"

    @property
    def nkpoints(self) -> int:
        return len(self.kpoints)

    @property
    def nbands(self) -> int:
        return self.num_bands // self.ispin

    @property
    def nelect(self) -> float:
        return self.num_electrons

    @property
    def spin_energies(self) -> np.ndarray:
        return self.energies.reshape(self.nkpoints, self.ispin, self.nbands).transpose(0, 2, 1)

    @property
    def spin_occupations(self) -> Optional[np.ndarray]:
        if self.occupations is None:
            return None
        return self.occupations.reshape(self.nkpoints, self.ispin, self.nbands).transpose(0, 2, 1)


H_SYMBOLS = {
    "GAMMA": "Γ", "G": "Γ", "GAM": "Γ", "SIGMA": "Σ",
    "DELTA": "Δ", "LAMBDA": "Λ",
}
H_SYMBOLS.update({symbol: symbol for symbol in "KMLXWAHRSTUYZD"})


def _format_label(raw_label: str) -> Optional[str]:
    label = raw_label.strip().strip("\"'")
    return H_SYMBOLS.get(label.upper(), label) if label else None


def _float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def _without_comment(line: str) -> str:
    return line.split("!", 1)[0].split("#", 1)[0].strip()


def _validate_lattice(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("lattice must be a finite 3x3 row matrix in angstrom")
    if np.linalg.cond(matrix) > 1e12:
        raise ValueError("lattice is singular or ill-conditioned")
    return matrix.copy()


class VASPLatticeParser:
    """Read only POSCAR's lattice header, including all scale conventions.

    One scalar (positive scale or negative target volume) supplies the VASP
    Cartesian KPOINTS conversion. Three positive Cartesian component scales
    calibrate the lattice but require an explicit KPOINTS conversion matrix.
    Atom positions are not needed for band-path distances.
    """

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)

    def parse(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        with self.filepath.open(encoding="utf-8-sig") as stream:
            lines = [stream.readline() for _ in range(5)]
        def error(line, message):
            return ValueError(f"{self.filepath}:{line}: {message}")
        if any(not line.strip() for line in lines[1:]):
            raise error(2, "incomplete POSCAR lattice header")
        try:
            scales = np.array([_float(v) for v in _without_comment(lines[1]).split()])
        except ValueError as exc:
            raise error(2, "invalid POSCAR scale") from exc
        if (len(scales) not in (1, 3) or not np.isfinite(scales).all() or
                (len(scales) == 1 and scales[0] == 0) or
                (len(scales) == 3 and np.any(scales <= 0))):
            raise error(2, "expected one nonzero scale or three positive scales")
        vectors = []
        for line_number, line in enumerate(lines[2:5], 3):
            try:
                vector = [_float(v) for v in _without_comment(line).split()]
            except ValueError as exc:
                raise error(line_number, "invalid lattice vector") from exc
            if len(vector) != 3 or not np.isfinite(vector).all():
                raise error(line_number, "expected three finite lattice components")
            vectors.append(vector)
        try:
            raw = _validate_lattice(vectors)
            if len(scales) == 1 and scales[0] < 0:
                scales = np.array([np.cbrt(-scales[0] / abs(np.linalg.det(raw)))])
            physical = _validate_lattice(raw * scales)
        except ValueError as exc:
            raise error(3, str(exc)) from exc
        # k_cart = (2π/s) x, B = (2π/s) inv(raw).T => f = x @ raw.T.
        conversion = raw.T.copy() if len(scales) == 1 else None
        return physical, conversion


def _extract_label_from_line(line: str, numeric_columns: int = 3) -> Optional[str]:
    positions = [line.index(marker) for marker in ("!", "#") if marker in line]
    if positions:
        return _format_label(line[min(positions) + 1:])
    return _format_label(" ".join(line.split()[numeric_columns:]))


def _equivalent_kpoints(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    delta = left - right
    return np.all(np.abs(delta - np.rint(delta)) <= 5e-7, axis=-1)


@dataclass
class KpointPath:
    points: np.ndarray
    labels: List[Tuple[int, str]]
    segments: List[Tuple[int, int]]
    coordinate_mode: str = "reciprocal"
    sampling_kind: str = "unknown"


class VASPKpointsParser:
    """Read endpoint pairs independently of optional blank lines.

    Each segment has N samples including both ends, as written by VASP.
    Automatic meshes have no path labels. Explicit lists may end in tetrahedra.
    """

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)

    def _error(self, line: int, message: str) -> ValueError:
        return ValueError(f"{self.filepath}:{line}: {message}")

    def parse(self) -> List[Tuple[int, str]]:
        return self.parse_path().labels

    def parse_path(
        self, expected_kpoints: Optional[np.ndarray] = None,
        cartesian_to_fractional: Optional[np.ndarray] = None,
    ) -> KpointPath:
        empty = KpointPath(np.empty((0, 3)), [], [])
        if not self.filepath.exists():
            return empty
        lines = self.filepath.read_text(encoding="utf-8-sig").splitlines()
        if len(lines) < 3:
            raise self._error(len(lines) + 1, "KPOINTS header is incomplete")
        try:
            count = int(_without_comment(lines[1]).split()[0])
        except (ValueError, IndexError) as exc:
            raise self._error(2, "expected a k-point count") from exc
        if count <= 0:
            if count == 0 and lines[2].strip().lower().startswith(('g', 'm')):
                if len(lines) < 4:
                    raise self._error(4, "automatic mesh needs three grid dimensions")
                try:
                    grid = list(map(int, _without_comment(lines[3]).split()))
                except ValueError as exc:
                    raise self._error(4, "automatic mesh dimensions must be integers") from exc
                if len(grid) != 3 or min(grid) < 1:
                    raise self._error(4, "automatic mesh needs three positive dimensions")
                empty.sampling_kind = "mesh"
            return empty
        line_mode = lines[2].strip().lower().startswith("l")
        coordinate_index = 3 if line_mode else 2
        if len(lines) <= coordinate_index or not _without_comment(lines[coordinate_index]):
            raise self._error(coordinate_index + 1, "missing coordinate mode")
        mode = "cartesian" if lines[coordinate_index].strip()[0].lower() in "ck" else "reciprocal"
        records = [(i + 1, line) for i, line in enumerate(lines[coordinate_index + 1:], coordinate_index + 1)
                   if _without_comment(line)]
        if line_mode:
            if count < 2 or not records or len(records) % 2:
                raise self._error(2, "line-mode needs N >= 2 and paired endpoints")
        else:
            if len(records) < count:
                raise self._error(len(lines) + 1, f"expected {count} explicit k points, got {len(records)}")
            records = records[:count]

        points, labels, segments, endpoints = [], [], [], []
        for line_number, line in records:
            tokens = _without_comment(line).split()
            numeric_columns = 3 if line_mode else 4
            try:
                values = [_float(token) for token in tokens[:numeric_columns]]
            except ValueError as exc:
                raise self._error(line_number, "invalid k-point coordinates or weight") from exc
            if len(values) != numeric_columns or not np.isfinite(values).all():
                raise self._error(line_number, "expected finite k-point coordinates and weight")
            endpoints.append((np.array(values[:3]), _extract_label_from_line(line, numeric_columns)))

        if line_mode:
            for pair in range(0, len(endpoints), 2):
                start, start_label = endpoints[pair]
                end, end_label = endpoints[pair + 1]
                first = len(points)
                points.extend(np.linspace(start, end, count))
                last = len(points) - 1
                segments.append((first, last))
                if start_label:
                    labels.append((first, start_label))
                if end_label:
                    labels.append((last, end_label))
        else:
            for index, (point, label) in enumerate(endpoints):
                points.append(point)
                if label:
                    labels.append((index, label))
        points = np.array(points)

        if expected_kpoints is not None:
            if mode == "cartesian":
                if cartesian_to_fractional is None:
                    raise self._error(coordinate_index + 1,
                                      "Cartesian KPOINTS requires a cartesian_to_fractional matrix to match EIGENVAL")
                matrix = np.asarray(cartesian_to_fractional, dtype=float)
                if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
                    raise ValueError("cartesian_to_fractional must be a finite invertible 3x3 matrix")
                points = points @ matrix
            if len(points) != len(expected_kpoints):
                raise self._error(2, f"KPOINTS expands to {len(points)} points but EIGENVAL has {len(expected_kpoints)}")
            matches = _equivalent_kpoints(points, expected_kpoints)
            if not matches.all():
                first_bad = int(np.flatnonzero(~matches)[0])
                raise self._error(coordinate_index + 2,
                                  f"KPOINTS coordinate mismatch at k-point {first_bad + 1}")
        return KpointPath(points, labels, segments, mode, "path" if line_mode else "explicit")


class VASPEigenvalParser:
    """Read standard NELECT/NKPTS/NBANDS with ISPIN=1 or 2.

    Old repository teaching files require explicit allow_legacy=True;
    the four-field header is never guessed to be standard VASP.
    """

    def __init__(
        self, filepath: str, kpoints_path: Optional[str] = None, *,
        allow_legacy: bool = False,
        cartesian_to_fractional: Optional[np.ndarray] = None,
        poscar_path: Optional[str] = None,
        lattice_matrix: Optional[np.ndarray] = None,
        cancel_check=None,
    ):
        self.filepath = Path(filepath)
        self.kpoints_path = Path(kpoints_path) if kpoints_path else None
        self.allow_legacy = allow_legacy
        self.cartesian_to_fractional = cartesian_to_fractional
        if poscar_path is not None and lattice_matrix is not None:
            raise ValueError("provide poscar_path or lattice_matrix, not both")
        self.poscar_path = Path(poscar_path) if poscar_path else None
        self.lattice_matrix = None if lattice_matrix is None else _validate_lattice(lattice_matrix)
        self.cancel_check = cancel_check

    def _error(self, line: int, message: str) -> ValueError:
        return ValueError(f"{self.filepath}:{line}: {message}")

    def parse(self) -> BandData:
        if self.cancel_check is not None:
            self.cancel_check()
        with self.filepath.open(encoding="utf-8-sig") as stream:
            points, weights, energies, occupations, nbands, nelect, ispin, legacy = self._read_stream(stream)
        return self._assemble_data(points, weights, energies, occupations, nbands, nelect, ispin, legacy)

    def _read_stream(self, stream):
        lines = []
        for _ in range(6):
            line = stream.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))
        if len(lines) < 6:
            raise self._error(len(lines) + 1, "EIGENVAL header is incomplete")
        header = lines[5].split()
        legacy = len(header) == 4
        if legacy and not self.allow_legacy:
            raise self._error(6, "nonstandard four-field header; use allow_legacy=True for old teaching files")
        if len(header) != 3 and not legacy:
            raise self._error(6, "expected NELECT NKPTS NBANDS")
        if legacy:
            try:
                if lines[0].strip() != "Generated example":
                    raise ValueError("unrecognized legacy header")
                _, _, nbands, nelect = map(int, header)
                ispin, nkpoints = 1, None
            except ValueError as exc:
                raise self._error(6, f"invalid legacy EIGENVAL header: {exc}") from exc
        else:
            try:
                ispin = int(lines[0].split()[-1])
            except (ValueError, IndexError) as exc:
                raise self._error(1, "invalid EIGENVAL header: expected ISPIN") from exc
            try:
                nelect, nkpoints, nbands = _float(header[0]), int(header[1]), int(header[2])
            except ValueError as exc:
                raise self._error(6, "invalid EIGENVAL header: expected numeric NELECT and integer NKPTS/NBANDS") from exc
        if ispin not in (1, 2):
            raise self._error(1, f"unsupported ISPIN={ispin}; expected 1 or 2")
        if not np.isfinite(nelect) or nelect < 0 or nbands <= 0 or (nkpoints is not None and nkpoints <= 0):
            raise self._error(6, "NELECT must be finite/nonnegative and NKPTS/NBANDS positive")
        if legacy:
            warnings.warn("Reading legacy teaching EIGENVAL; NKPTS was not declared and old KPOINTS is ignored",
                          UserWarning, stacklevel=2)

        # Standard files declare dimensions; keep only the output arrays and one line.
        if nkpoints is not None:
            points = np.empty((nkpoints, 3))
            weights = np.empty(nkpoints)
            energies = np.empty((nkpoints, nbands * ispin))
            occupations = np.empty_like(energies)
        else:
            points, weights, energies, occupations = [], [], [], []
        line_number, point_index = 6, 0
        while True:
            if self.cancel_check is not None:
                self.cancel_check()
            line = stream.readline()
            line_number += 1
            while line and not line.strip():
                line = stream.readline()
                line_number += 1
            if not line:
                if nkpoints is not None and point_index != nkpoints:
                    raise self._error(line_number, f"truncated EIGENVAL: expected {nkpoints} k points, got {point_index}")
                break
            if nkpoints is not None and point_index == nkpoints:
                raise self._error(line_number, f"unexpected data after {nkpoints} declared k points")
            try:
                k_values = [_float(token) for token in line.split()]
            except ValueError as exc:
                raise self._error(line_number, "invalid k-point coordinates/weight") from exc
            if len(k_values) != 4 or not np.isfinite(k_values).all() or k_values[3] < 0:
                raise self._error(line_number, "expected 3 finite coordinates and a nonnegative finite weight")
            if nkpoints is None:
                points.append(k_values[:3])
                weights.append(k_values[3])
                row_energies = np.empty(nbands * ispin)
                row_occupations = np.empty_like(row_energies)
            else:
                points[point_index] = k_values[:3]
                weights[point_index] = k_values[3]
                row_energies = energies[point_index]
                row_occupations = occupations[point_index]
            for band in range(nbands):
                if band % 128 == 0 and self.cancel_check is not None:
                    self.cancel_check()
                line = stream.readline()
                line_number += 1
                if not line:
                    raise self._error(line_number, f"truncated k-point {point_index + 1}, expected band {band + 1}/{nbands}")
                fields = line.split()
                if len(fields) != 1 + 2 * ispin:
                    raise self._error(line_number, f"k-point {point_index + 1}, band {band + 1}: expected {1 + 2 * ispin} columns for ISPIN={ispin}")
                try:
                    ordinal = int(fields[0])
                    values = [_float(token) for token in fields[1:]]
                except ValueError as exc:
                    raise self._error(line_number, f"invalid band row at k-point {point_index + 1}, band {band + 1}") from exc
                if ordinal != band + 1 or not np.isfinite(values).all():
                    raise self._error(line_number, f"expected band index {band + 1} and finite energies/occupations")
                for channel in range(ispin):
                    row_energies[channel * nbands + band] = values[channel]
                    row_occupations[channel * nbands + band] = values[ispin + channel]
            if nkpoints is None:
                energies.append(row_energies)
                occupations.append(row_occupations)
            point_index += 1
        if point_index == 0:
            raise self._error(7, "no k points found")
        return points, weights, energies, occupations, nbands, nelect, ispin, legacy

    def _assemble_data(self, points, weights, energies, occupations, nbands, nelect, ispin, legacy):
        points = np.asarray(points)
        path_file = self.kpoints_path if self.kpoints_path is not None else self.filepath.parent / "KPOINTS"
        if self.kpoints_path is not None and not path_file.exists():
            raise FileNotFoundError(f"KPOINTS file not found: {path_file}")
        path = KpointPath(np.empty((0, 3)), [], [])
        lattice, source = self.lattice_matrix, None
        conversion = self.cartesian_to_fractional
        if lattice is not None:
            source = "explicit matrix"
        else:
            poscar = self.poscar_path or self.filepath.parent / "POSCAR"
            if self.poscar_path is not None and not poscar.exists():
                raise FileNotFoundError(f"POSCAR file not found: {poscar}")
            if poscar.exists():
                lattice, automatic_conversion = VASPLatticeParser(str(poscar)).parse()
                source = str(poscar.resolve())
                if conversion is None:
                    conversion = automatic_conversion
        if not legacy:
            path = VASPKpointsParser(str(path_file)).parse_path(points, conversion)
        distance_points = points if lattice is None else points @ (2 * np.pi * np.linalg.inv(lattice).T)
        return BandData(
            kpoints=points, kdistances=self._calc_kdistances(distance_points, path.segments),
            energies=np.asarray(energies), num_bands=nbands * ispin, num_electrons=nelect,
            kpoint_labels=path.labels, ispin=ispin, occupations=np.asarray(occupations),
            weights=np.asarray(weights), segments=path.segments,
            input_format="legacy-teaching" if legacy else "vasp",
            lattice_matrix=lattice, lattice_source=source,
            sampling_kind=path.sampling_kind,
        )

    @staticmethod
    def _calc_kdistances(kpoints: np.ndarray, segments=()) -> np.ndarray:
        steps = np.linalg.norm(np.diff(kpoints, axis=0), axis=1)
        for start, _ in segments[1:]:
            steps[start - 1] = 0.0
        return np.r_[0.0, np.cumsum(steps)]
