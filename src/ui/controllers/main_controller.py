"""Transactional GUI controller: compute candidate state before publishing it."""
from copy import deepcopy
from PyQt5.QtCore import QObject, pyqtSignal
from models.simulation_state import SimulationState, CrystalOrientation
from services.simulation_service import SimulationService
from point_groups import get_components_for_group, create_tensor_with_relations


class MainController(QObject):
    state_changed = pyqtSignal()
    plot_data_ready = pyqtSignal(object, object)
    components_updated = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._state = SimulationState()
        self._simulation_service = SimulationService()

    @property
    def state(self):
        return self._state

    def _change(self, change, components=False):
        candidate = deepcopy(self._state)
        try:
            change(candidate)
            x, y = self._simulation_service.compute_shg_intensity(candidate)
        except Exception as error:
            self.error_occurred.emit(str(error))
            self.state_changed.emit()
            return False
        self._state = candidate
        if components:
            self.components_updated.emit(list(candidate.component_values))
        self.plot_data_ready.emit(x, y)
        self.state_changed.emit()
        return True

    def set_point_group(self, group):
        def change(s):
            s.point_group_name = group
            s.component_values = {c: 1.0 for c in get_components_for_group(group)}
            s.base_tensor = create_tensor_with_relations(group)
        return self._change(change, components=True)

    def set_component_value(self, component, value):
        def change(s):
            if component not in s.component_values:
                raise ValueError('Unknown tensor parameter')
            s.component_values[component] = value
        return self._change(change)

    def set_global_scale(self, value):
        return self._change(lambda s: setattr(s, 'global_scale', value))

    def set_orientation(self, phi, theta, psi):
        return self._change(lambda s: setattr(s, 'orientation', CrystalOrientation(phi, theta, psi)))

    def set_parameter(self, name, value):
        if name not in ('scan_mode', 'phi_beam', 'theta_fixed', 'alpha_polarization', 'detection_mode', 'input_polarization_mode'):
            raise ValueError(name)
        return self._change(lambda s: setattr(s, name, value))

    def reset_parameters(self):
        def change(s):
            for c in s.component_values:
                s.component_values[c] = 1.0
            s.global_scale = 1.0
            s.orientation = CrystalOrientation()
            s.phi_beam = s.theta_fixed = s.alpha_polarization = 0
            s.scan_mode = 'alpha_polar'
            s.detection_mode = 0
            s.input_polarization_mode = 'linear'
        return self._change(change)

    def _trigger_recalculation(self):
        return self._change(lambda s: None)
