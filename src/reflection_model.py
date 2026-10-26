"""Relative reflected SHG from an in-plane nonlinear sheet on an isotropic stack.

e^(-i wt), passive n+ik. Tangential Fresnel coefficients and reciprocal local
fields are included at w and 2w. A common intensity prefactor is fitted;
this module does not infer an absolute susceptibility from detector counts.
"""
from dataclasses import dataclass, asdict
import numpy as np
from point_groups import get_tensor_basis
from core.tensor_operations import TensorOperations


@dataclass(frozen=True)
class Geometry:
    incidence_deg: float = 0.0
    beam_azimuth_deg: float = 0.0
    polarization_zero_deg: float = 0.0
    analyzer_mode: str = 'co_rotating'
    analyzer_offset_deg: float = 0.0
    fixed_analyzer_deg: float = 0.0
    perpendicular_gain: float = 1.0
    gain_calibrated: bool = False
    zero_calibrated: bool = False

    def __post_init__(self):
        numbers = [self.incidence_deg, self.beam_azimuth_deg, self.polarization_zero_deg,
                   self.analyzer_offset_deg, self.fixed_analyzer_deg, self.perpendicular_gain]
        if not np.isfinite(numbers).all() or not 0 <= self.incidence_deg < 85 or self.perpendicular_gain <= 0:
            raise ValueError('几何参数需有限，入射角应在 [0,85)°，通道响应比必须为正。')
        if self.analyzer_mode not in ('co_rotating', 'fixed'):
            raise ValueError('请选择检偏器是否同步旋转。')


@dataclass(frozen=True)
class OpticalStack:
    mode: str = 'relative'
    pump_nm: float = None
    substrate_omega: complex = 1+0j
    substrate_2omega: complex = 1+0j
    film_omega: complex = 1+0j
    film_2omega: complex = 1+0j
    film_thickness_nm: float = 0.0
    constants_source: str = ''

    def __post_init__(self):
        if self.mode not in ('relative', 'single', 'film'):
            raise ValueError('未知基底光学模型。')
        if not np.isfinite(self.film_thickness_nm) or self.film_thickness_nm < 0 or (self.pump_nm is not None and (not np.isfinite(self.pump_nm) or self.pump_nm <= 0)):
            raise ValueError('波长必须为正，膜厚必须非负。')
        if self.mode != 'relative' and self.pump_nm is None:
            raise ValueError('反射修正需要填写实际激发波长。')
        for value in (self.substrate_omega, self.substrate_2omega, self.film_omega, self.film_2omega):
            value = complex(value)
            if not np.isfinite(value) or value.real <= 0 or value.imag < 0:
                raise ValueError('折射率使用被动介质 n+ik：n>0，k>=0。')
        if self.mode != 'relative' and not self.constants_source.strip():
            raise ValueError('请填写折射率来源或明确注明为估计值。')

    def to_dict(self):
        result = asdict(self)
        for key in ('substrate_omega', 'substrate_2omega', 'film_omega', 'film_2omega'):
            result[key] = str(result[key])
        return result

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        for key in ('substrate_omega', 'substrate_2omega', 'film_omega', 'film_2omega'):
            if key in values:
                values[key] = complex(values[key])
        return cls(**values)


def tangential_reflection(indices, thickness_nm, wavelength_nm, incidence_deg, polarization):
    """Stable passive-stack recursion; reflection of tangential E, not conventional signed rp."""
    n = np.asarray(indices, dtype=complex)
    kz = np.sqrt(n*n - np.sin(np.deg2rad(incidence_deg))**2 + 0j)
    kz = np.where((kz.imag < 0) | ((np.abs(kz.imag) < 1e-14) & (kz.real < 0)), -kz, kz)
    admittance = kz if polarization == 's' else n*n/kz
    r = (admittance[-2] - admittance[-1]) / (admittance[-2] + admittance[-1])
    for j in range(len(n)-3, -1, -1):
        interface = (admittance[j] - admittance[j+1]) / (admittance[j] + admittance[j+1])
        phase = np.exp(4j*np.pi*kz[j+1]*thickness_nm[j]/wavelength_nm)
        r = (interface + r*phase)/(1 + interface*r*phase)
    return r


def local_factors(stack, geometry):
    cosine = np.cos(np.deg2rad(geometry.incidence_deg))
    if stack.mode == 'relative':
        return np.array([cosine, 1], complex), np.array([cosine, 1], complex)
    factors = []
    for wavelength, substrate, film in [(stack.pump_nm, stack.substrate_omega, stack.film_omega),
                                         (stack.pump_nm/2, stack.substrate_2omega, stack.film_2omega)]:
        indices = [1, substrate] if stack.mode == 'single' else [1, film, substrate]
        thickness = [] if stack.mode == 'single' else [stack.film_thickness_nm]
        rp = tangential_reflection(indices, thickness, wavelength, geometry.incidence_deg, 'p')
        rs = tangential_reflection(indices, thickness, wavelength, geometry.incidence_deg, 's')
        factors.append(np.array([cosine*(1+rp), 1+rs]))
    return factors[0], factors[1]


def field_basis(group, angles_deg, orientation_deg, geometry=None, stack=None, channel='parallel'):
    geometry, stack = geometry or Geometry(), stack or OpticalStack()
    basis = get_tensor_basis(group)
    names = list(basis)
    if not names:
        return names, np.zeros((len(angles_deg), 0), complex)
    R = TensorOperations.get_rotation_matrix(orientation_deg, 0, 0)
    blocks = np.einsum('pi,qj,rk,bijk->bpqr', R, R, R, np.asarray(list(basis.values())))
    angle = np.deg2rad(np.asarray(angles_deg) + geometry.polarization_zero_deg)
    phi = np.deg2rad(geometry.beam_azimuth_deg)
    p = np.array([np.cos(phi), np.sin(phi), 0])
    s = np.array([-np.sin(phi), np.cos(phi), 0])
    incoming, outgoing = local_factors(stack, geometry)
    E = np.cos(angle)[:, None]*incoming[0]*p + np.sin(angle)[:, None]*incoming[1]*s
    P = np.einsum('bijk,nj,nk->nbi', blocks, E, E)
    Ep, Es = np.einsum('nbi,i->nb', P, p)*outgoing[0], np.einsum('nbi,i->nb', P, s)*outgoing[1]
    analyzer = angle if geometry.analyzer_mode == 'co_rotating' else np.full_like(angle, np.deg2rad(geometry.fixed_analyzer_deg))
    analyzer = analyzer + np.deg2rad(geometry.analyzer_offset_deg)
    if channel == 'perpendicular':
        return names, -np.sin(analyzer)[:, None]*Ep + np.cos(analyzer)[:, None]*Es
    if channel != 'parallel':
        raise ValueError('未知检测通道。')
    return names, np.cos(analyzer)[:, None]*Ep + np.sin(analyzer)[:, None]*Es


def predict(group, angles_deg, orientation_deg, values, geometry=None, stack=None, channel='parallel'):
    geometry = geometry or Geometry()
    names, fields = field_basis(group, angles_deg, orientation_deg, geometry, stack, channel)
    coefficients = np.array([complex(values.get(name, 0)) for name in names])
    gain = geometry.perpendicular_gain if channel == 'perpendicular' else 1
    return gain*np.abs(fields @ coefficients)**2


def observable_parameters(group, geometry=None, stack=None):
    names, fields = field_basis(group, np.arange(0, 360, 7.5), 0, geometry, stack)
    _, crossed = field_basis(group, np.arange(0, 360, 7.5), 0, geometry, stack, 'perpendicular')
    return [name for i, name in enumerate(names) if max(np.max(np.abs(fields[:, i])), np.max(np.abs(crossed[:, i]))) > 1e-10]
