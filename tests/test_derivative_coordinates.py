"""Independent Cartesian polynomials test reciprocal-row derivative conventions."""

import numpy as np
import pytest

from vdw_studio.analysis.properties import effective_mass, principal_masses, fermi_velocity
from vdw_studio.engine.models import HoneycombModel
from vdw_studio.structure.lattice import Lattice


class CartesianPolynomial:
    def __init__(self, rotation=None, offset=2.):
        rotation = np.eye(3) if rotation is None else rotation
        a = 2.5
        raw = np.array([[a, 0, 0], [-a / 2, np.sqrt(3) * a / 2, 0], [0, 0, 15]])
        self.lattice = Lattice(raw @ rotation)
        self.center = np.array([.17, -.2])
        self.rotation = rotation
        self.offset = offset

    def energies_at(self, k):
        q = ((np.asarray(k) - self.center) @ self.lattice.reciprocal_matrix[:2]) @ self.rotation.T
        # Hessian eigenvalues 2 and 10 eV Å², with no fractional-coordinate formula.
        return np.array([self.offset + q[0] ** 2 + 5 * q[1] ** 2])


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def test_hexagonal_fractional_direction_uses_physical_metric():
    model = CartesianPolynomial()
    # b1 points 30 degrees from x: curvature = 2*(3/4) + 10*(1/4) = 4.
    mass = effective_mass(model, model.center, 0, direction=(1, 0))
    assert mass == pytest.approx(7.619964 / 4, rel=1e-7)


def test_cartesian_direction_has_explicit_physical_step():
    model = CartesianPolynomial()
    assert effective_mass(model, model.center, 0, direction=(1, 0),
                          direction_space="cartesian") == pytest.approx(3.809982, rel=1e-7)
    assert effective_mass(model, model.center, 0, direction=(0, 1, 0),
                          direction_space="cartesian") == pytest.approx(.7619964, rel=1e-7)


@pytest.mark.parametrize("rotation", [np.eye(3), rotation_z(.53),
    np.array([[1, 0, 0], [0, .6, -.8], [0, .8, .6]]),
    np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])])
def test_principal_masses_survive_nonorthogonal_and_rotated_planes(rotation):
    model = CartesianPolynomial(rotation)
    masses = principal_masses(model, model.center, 0)[:2]
    assert sorted(masses) == pytest.approx([.7619964, 3.809982], rel=1e-7)


def test_principal_axis_angle_uses_cartesian_plane():
    model = CartesianPolynomial(rotation_z(.53))
    theta = principal_masses(model, model.center, 0)[2]
    assert abs(np.cos(theta + .53)) == pytest.approx(1., abs=1e-7)


@pytest.mark.parametrize("offset", [0., 7.3, -12.5])
def test_dirac_velocity_is_independent_of_energy_zero(offset):
    original = HoneycombModel(a=2.46, t=-2.7)
    class Shifted:
        lattice = original.lattice
        def energies_at(self, k):
            return original.energies_at(k) + offset
    speed = fermi_velocity(Shifted(), (1/3, 1/3), 1, direction=(1, -1))
    # Independent nearest-neighbor Dirac cone slope √3 |t| a / 2 (eV Å).
    assert speed == pytest.approx(np.sqrt(3) * 2.7 * 2.46 / 2 * 1.51927e5, rel=2e-5)


def test_tilted_plane_speed_and_fractional_cartesian_directions_agree():
    rotation = np.array([[1, 0, 0], [0, .6, -.8], [0, .8, .6]])
    model = CartesianPolynomial(rotation)
    axis = np.array([np.sqrt(3) / 2, .5, 0]) @ rotation
    physical_step = 1e-4 * 4 * np.pi / (np.sqrt(3) * 2.5)
    mass = effective_mass(model, model.center, 0)
    cart_mass = effective_mass(model, model.center, 0, direction=axis,
                               dk=physical_step, direction_space="cartesian")
    assert mass == pytest.approx(cart_mass, rel=1e-8)


def test_out_of_plane_direction_is_rejected():
    model = CartesianPolynomial()
    with pytest.raises(ValueError, match="reciprocal plane"):
        effective_mass(model, model.center, 0, direction=(0, 0, 1), direction_space="cartesian")


@pytest.mark.parametrize("kwargs", [{"dk": 0}, {"dk": float('nan')},
    {"direction": (0, 0)}, {"direction": (1, float('nan'))}, {"band": -1}])
def test_invalid_derivative_inputs_are_rejected(kwargs):
    model = CartesianPolynomial()
    options = {"band": 0}
    options.update(kwargs)
    with pytest.raises(ValueError):
        effective_mass(model, model.center, **options)


def test_flat_curvature_returns_infinite_mass():
    class Flat(CartesianPolynomial):
        def energies_at(self, k):
            return np.array([5.])
    model = Flat()
    assert effective_mass(model, model.center, 0) == float('inf')
    assert principal_masses(model, model.center, 0)[:2] == (float('inf'), float('inf'))
