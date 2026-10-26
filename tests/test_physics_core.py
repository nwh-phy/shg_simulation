import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
import sympy as sp
from scipy.spatial.transform import Rotation
from point_groups import get_all_point_groups, get_tensor_basis, create_tensor_with_relations, point_group_components
from core.tensor_operations import TensorOperations as T
from core.shg_calculator import SHGCalculator, ScanParameters, PolarizationMode, DetectionMode
from symbolic_calculator import calculate_symbolic_shg_expressions, alpha, E0
from reflection_model import Geometry, OpticalStack, local_factors, predict, tangential_reflection


@pytest.mark.parametrize('group', get_all_point_groups())
def test_all_independent_basis_tensors_obey_group_and_shg_symmetry(group):
    for tensor in get_tensor_basis(group).values():
        np.testing.assert_allclose(tensor, tensor.swapaxes(1, 2))
        for R in point_group_components[group]['Generators']:
            R = np.array(R)
            np.testing.assert_allclose(T.rotate_tensor_vectorized(tensor, R), tensor, atol=1e-12)


def test_euler_convention_and_full_contraction_voigt_agree():
    R = T.get_rotation_matrix(23, 47, 61)
    np.testing.assert_allclose(R, Rotation.from_euler('ZYZ', [23, 47, 61], degrees=True).as_matrix(), atol=1e-14)
    rng = np.random.default_rng(182)
    # Also test nonsymmetric integer inputs: Voigt must use the symmetric part.
    tensor = rng.integers(-3, 4, (3, 3, 3))
    E = rng.normal(size=3)
    vector = [E[0]**2, E[1]**2, E[2]**2, 2*E[1]*E[2], 2*E[0]*E[2], 2*E[0]*E[1]]
    np.testing.assert_allclose(T.tensor_to_voigt(tensor)@vector, np.einsum('ijk,j,k->i', tensor, E, E), atol=1e-13)
    group = next(g for g in get_all_point_groups() if g.startswith('432 ='))
    assert not np.any(create_tensor_with_relations(group))


def test_normal_incidence_d3h_sixfold_and_channel_sum():
    group = next(g for g in get_all_point_groups() if g.startswith('6̄m2 ='))
    a = np.linspace(0, 2*np.pi, 73)
    calculator = SHGCalculator(T.tensor_to_voigt(create_tensor_with_relations(group)))
    par = calculator.calculate_intensity_vectorized(ScanParameters(np.zeros_like(a), alpha_fixed=a, input_polarization=PolarizationMode.LINEAR, detection_mode=DetectionMode.PARALLEL))
    perp = calculator.calculate_intensity_vectorized(ScanParameters(np.zeros_like(a), alpha_fixed=a, input_polarization=PolarizationMode.LINEAR, detection_mode=DetectionMode.PERPENDICULAR))
    np.testing.assert_allclose(par, np.sin(3*a)**2, atol=3e-14)
    np.testing.assert_allclose(par+perp, 1, atol=1e-14)
    np.testing.assert_allclose(predict(group, np.rad2deg(a), 0, {'yyy': 1}), par, atol=1e-14)


def test_symbolic_numeric_agreement_after_nontrivial_orientation():
    group = next(g for g in get_all_point_groups() if g.startswith('3m ='))
    values = dict(zip(get_tensor_basis(group), [0.7, 1.2, -0.3, 0.9]))
    orientation = dict(phi_c=23, theta_c=31, psi_c=17)
    expressions = calculate_symbolic_shg_expressions(group, orientation)
    substitutes = {sp.Symbol('d_'+name, real=True): value for name, value in values.items()}
    substitutes.update({alpha: 0.41, E0: 1})
    tensor = T.rotate_tensor_vectorized(create_tensor_with_relations(group, values=values), T.get_rotation_matrix(23, 31, 17))
    E = np.array([np.cos(0.41), np.sin(0.41), 0])
    P = np.einsum('ijk,j,k->i', tensor, E, E)
    for key, value in zip(['Px', 'Py', 'Pz'], P):
        assert float(expressions[key].subs(substitutes)) == pytest.approx(value, abs=1e-12)
    assert float(expressions['I_total'].subs(substitutes)) == pytest.approx(P[0]**2+P[1]**2)


def test_fresnel_single_interface_zero_thickness_and_passive_stack():
    for polarization in ['s', 'p']:
        r = tangential_reflection([1, 1.5], [], 1000, 0, polarization)
        assert r == pytest.approx(-0.2)
        film = tangential_reflection([1, 2.1, 1.5], [0], 1000, 0, polarization)
        assert film == pytest.approx(r)
        for angle in [0, 15, 50, 80]:
            assert abs(tangential_reflection([1, 1.46, 4+0.2j], [300], 532, angle, polarization)) <= 1+1e-12
    stack = OpticalStack('single', 1064, 1.5, 1.6, constants_source='synthetic test indices')
    incoming, outgoing = local_factors(stack, Geometry())
    np.testing.assert_allclose(incoming, [0.8, 0.8])
    np.testing.assert_allclose(outgoing, [2/2.6, 2/2.6])


def test_freestanding_reflection_matches_relative_model():
    group = next(g for g in get_all_point_groups() if g.startswith('6̄m2 ='))
    stack = OpticalStack('single', 1064, constants_source='synthetic freestanding n=1')
    angles = np.arange(0, 360, 12)
    for channel in ['parallel', 'perpendicular']:
        np.testing.assert_allclose(predict(group, angles, 18, {'yyy': 1}, Geometry(incidence_deg=25), stack, channel), predict(group, angles, 18, {'yyy': 1}, Geometry(incidence_deg=25), channel=channel))
