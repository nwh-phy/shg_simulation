"""Vectorized normalized SHG polarization and transverse detector projection."""
from dataclasses import dataclass
from enum import Enum
import numpy as np


class PolarizationMode(Enum):
    DEFAULT = 'default'
    LINEAR = 'linear'
    LCP = 'lcp'
    RCP = 'rcp'


class DetectionMode(Enum):
    TOTAL = 0
    PARALLEL = 1
    PERPENDICULAR = 2


@dataclass
class ScanParameters:
    theta_range: np.ndarray
    phi_fixed: float = 0.0
    alpha_fixed: float = 0.0
    input_polarization: PolarizationMode = PolarizationMode.DEFAULT
    detection_mode: DetectionMode = DetectionMode.TOTAL


class SHGCalculator:
    def __init__(self, d_matrix):
        self.d_matrix = np.asarray(d_matrix)
        if self.d_matrix.shape != (3, 6) or not np.isfinite(self.d_matrix).all():
            raise ValueError('d matrix must be finite and have shape (3,6).')

    def _compute_electric_field_vectorized(self, theta, phi, alpha, mode):
        theta, alpha = np.broadcast_arrays(np.atleast_1d(theta), np.asarray(alpha))
        p = np.column_stack([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)])
        s = np.broadcast_to([-np.sin(phi), np.cos(phi), 0.0], p.shape)
        if mode == PolarizationMode.LCP:
            return (p + 1j*s)/np.sqrt(2)
        if mode == PolarizationMode.RCP:
            return (p - 1j*s)/np.sqrt(2)
        if mode == PolarizationMode.LINEAR:
            return np.cos(alpha)[:, None]*p + np.sin(alpha)[:, None]*s
        return p

    def _compute_voigt_vector_vectorized(self, E):
        x, y, z = E.T
        return np.column_stack([x*x, y*y, z*z, 2*y*z, 2*x*z, 2*x*y])

    def calculate_intensity_vectorized(self, params):
        theta = np.atleast_1d(params.theta_range)
        E = self._compute_electric_field_vectorized(theta, params.phi_fixed, params.alpha_fixed, params.input_polarization)
        P = np.einsum('il,nl->ni', self.d_matrix, self._compute_voigt_vector_vectorized(E))
        k = np.column_stack([np.sin(theta)*np.cos(params.phi_fixed), np.sin(theta)*np.sin(params.phi_fixed), np.cos(theta)])
        transverse = P - np.sum(P*k, axis=1)[:, None]*k
        if params.detection_mode == DetectionMode.TOTAL:
            return np.sum(np.abs(transverse)**2, axis=1)
        if params.input_polarization in (PolarizationMode.LCP, PolarizationMode.RCP):
            raise ValueError('Parallel/perpendicular linear analysis is undefined for circular input; select total or linear input.')
        analyzer = E if params.detection_mode == DetectionMode.PARALLEL else np.cross(k, E)
        return np.abs(np.sum(transverse*np.conjugate(analyzer), axis=1))**2
