"""Symbolic and numeric tensors share the symmetry catalog and ZYZ convention."""
import sympy as sp
import numpy as np
from point_groups import get_tensor_basis
from core.tensor_operations import TensorOperations

alpha = sp.symbols('alpha', real=True)
E0 = sp.symbols('E0', real=True)


def get_symbolic_d_tensor(point_group_name):
    tensor = sp.MutableDenseNDimArray.zeros(3, 3, 3)
    for name, basis in get_tensor_basis(point_group_name).items():
        parameter = sp.Symbol('d_' + name, real=True)
        for index in np.ndindex(3, 3, 3):
            if basis[index]:
                tensor[index] += sp.nsimplify(basis[index]) * parameter
    return tensor


def calculate_symbolic_shg_expressions(point_group_name, crystal_orientation_info=None):
    tensor = get_symbolic_d_tensor(point_group_name)
    orientation = crystal_orientation_info or {}
    rotation = TensorOperations.get_rotation_matrix(*(orientation.get(k, 0) for k in ['phi_c', 'theta_c', 'psi_c']))
    # Form the rotated tensor via the same active rotation used by numeric calculations.
    rotated = sp.MutableDenseNDimArray.zeros(3, 3, 3)
    for p, q, r in np.ndindex(3, 3, 3):
        rotated[p, q, r] = sum(sp.Float(rotation[p, i]*rotation[q, j]*rotation[r, k], 15)*tensor[i, j, k]
                               for i, j, k in np.ndindex(3, 3, 3) if rotation[p, i]*rotation[q, j]*rotation[r, k] != 0)
    electric = [E0*sp.cos(alpha), E0*sp.sin(alpha), sp.S.Zero]
    polarization = [sp.expand(sum(rotated[i, j, k]*electric[j]*electric[k] for j in range(3) for k in range(3))) for i in range(3)]
    voigt = sp.Matrix([[rotated[i, j, k] for j, k in [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]] for i in range(3)])
    parallel = polarization[0]*sp.cos(alpha) + polarization[1]*sp.sin(alpha)
    perpendicular = -polarization[0]*sp.sin(alpha) + polarization[1]*sp.cos(alpha)
    return {'d_tensor_str': str(voigt), 'd_voigt_matrix_obj': voigt,
            'Px': polarization[0], 'Py': polarization[1], 'Pz': polarization[2],
            'I_total': polarization[0]**2 + polarization[1]**2, 'I_parallel': sp.expand(parallel**2), 'I_perpendicular': sp.expand(perpendicular**2)}
