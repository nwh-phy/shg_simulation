"""Symmetry-safe simulation service shared by the simulator views."""
import numpy as np
from core.tensor_operations import TensorOperations
from core.shg_calculator import SHGCalculator, ScanParameters, PolarizationMode, DetectionMode
from point_groups import create_tensor_with_relations


class SimulationService:
    def _get_scaled_tensor(self, state):
        if state.point_group_name:
            tensor = create_tensor_with_relations(state.point_group_name, values=state.component_values)
        else:
            tensor = np.zeros((3, 3, 3)) if state.base_tensor is None else state.base_tensor.copy()
        return tensor*state.global_scale

    def compute_lab_frame_d_matrix(self, state):
        tensor = self._get_scaled_tensor(state)
        o = state.orientation
        R = TensorOperations.get_rotation_matrix(o.phi_c, o.theta_c, o.psi_c)
        return TensorOperations.tensor_to_voigt(TensorOperations.rotate_tensor_vectorized(tensor, R))

    def _build_scan_parameters(self, state):
        scan = np.linspace(0, 2*np.pi, 361)
        alpha_scan = state.scan_mode.startswith('alpha')
        return ScanParameters(theta_range=np.full_like(scan, np.deg2rad(state.theta_fixed)) if alpha_scan else scan,
                              phi_fixed=np.deg2rad(state.phi_beam),
                              alpha_fixed=scan if alpha_scan else np.deg2rad(state.alpha_polarization),
                              input_polarization=PolarizationMode.LINEAR if alpha_scan else PolarizationMode(state.input_polarization_mode),
                              detection_mode=DetectionMode(state.detection_mode))

    def compute_shg_intensity(self, state):
        parameters = self._build_scan_parameters(state)
        curve = SHGCalculator(self.compute_lab_frame_d_matrix(state)).calculate_intensity_vectorized(parameters)
        x = parameters.alpha_fixed if state.scan_mode.startswith('alpha') else parameters.theta_range
        return x, curve
