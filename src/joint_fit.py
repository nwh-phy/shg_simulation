"""Joint intensity fitting with shared tensor/orientation and explicit identifiability."""
from dataclasses import dataclass, asdict
from copy import deepcopy
import numpy as np
from scipy.optimize import least_squares
from experimental_data import ExperimentalCurve
from point_groups import get_components_for_group
from reflection_model import Geometry, OpticalStack, predict, observable_parameters


@dataclass(frozen=True)
class FitOptions:
    fit_ratios: bool = False
    complex_ratios: bool = False
    fit_background: bool = True
    fit_orientation: bool = True
    fixed_orientation_deg: float = 0.0
    starts: int = 12
    bootstrap: int = 100
    seed: int = 20261026
    ratio_bound: float = 10.0
    reference_component: str = ''

    def __post_init__(self):
        if not 1 <= self.starts <= 72 or not 0 <= self.bootstrap <= 1000:
            raise ValueError('多起点数量应为 1–72，重采样数量为 0–1000。')
        if not np.isfinite(self.fixed_orientation_deg):
            raise ValueError('固定取向必须有限。')
        if not np.isfinite(self.ratio_bound) or self.ratio_bound <= 0:
            raise ValueError('张量比上限必须为有限正数。')


def orientation_period(group):
    symbol = group.split(' =')[0]
    return 60.0 if symbol in ('3', '32', '3m', '6̄', '6̄m2') else 180.0


def _uncertainties(curve):
    records = curve.metadata.get('records', [])
    if len(records) == len(curve.intensity):
        errors = [r.get('uncertainty') for r in records]
        if all(e is not None and np.isfinite(e) and e > 0 for e in errors):
            return np.asarray(errors)
    return None


def fit_joint(curves, group, geometry=None, stack=None, options=None, tensor_values=None, cancelled=None, progress=None):
    geometry, stack, options = geometry or Geometry(), stack or OpticalStack(), options or FitOptions()
    warnings, channels, angles, ys, sigmas, masks = [], [], [], [], [], []
    if set(curves) != {'parallel', 'perpendicular'}:
        raise ValueError('联合拟合需要平行和垂直两组数据。请先载入两个通道。')
    quantities = set()
    for channel in ('parallel', 'perpendicular'):
        curve = curves[channel]
        if np.ndim(curve.angle_deg) != 1 or len(curve.angle_deg) != len(curve.intensity) or not np.isfinite(curve.angle_deg).all() or not np.isfinite(curve.intensity).all():
            raise ValueError('输入角度和强度需为等长的一维有限数组。')
        mask = np.ones(len(curve.angle_deg), dtype=bool)
        excluded = curve.metadata.get('excluded_indices', [])
        if any(not isinstance(i, int) or i < 0 or i >= len(mask) for i in excluded):
            raise ValueError('排除点索引无效。')
        mask[excluded] = False
        records = curve.metadata.get('records', [])
        if len(records) == len(mask) and any('saturated' in r.get('quality_flags', []) and mask[i] for i, r in enumerate(records)):
            raise ValueError('含未排除的饱和测量。请在测量质量表中明确排除，或重新采集。')
        if sum(mask) < 4:
            raise ValueError('每个通道至少需要四个有效测量点。')
        method = curve.metadata.get('extraction', {}).get('method')
        if method:
            quantities.add('area' if method == 'integral' else 'height')
        channels.append(channel)
        masks.append(mask)
        angles.append(np.asarray(curve.angle_deg)[mask])
        ys.append(np.asarray(curve.intensity)[mask])
        errors = _uncertainties(curve)
        sigmas.append(errors[mask] if errors is not None else None)
    if len(quantities) > 1:
        raise ValueError('两个通道分别使用面积和点/峰高，物理量不一致，请用同一种提取方式。')
    units = {c.metadata.get('intensity_unit', 'input_units') for c in curves.values()}
    if len(units) > 1:
        raise ValueError('两个通道的强度单位不同，请统一曝光/功率归一化和提取方式。')
    labels = get_components_for_group(group)
    visible = observable_parameters(group, geometry, stack)
    if not visible:
        raise ValueError('此点群在当前面内薄片几何下没有可观察的电偶极信号。请核查材料、点群和模型适用范围。')
    values = {name: complex((tensor_values or {}).get(name, 1.0)) for name in labels}
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError('张量参数必须有限。')
    if options.reference_component and options.reference_component not in visible:
        raise ValueError('参考分量必须是在当前几何下可观测的独立分量。')
    reference = options.reference_component or next((name for name in visible if abs(values[name]) > 1e-10), visible[0])
    if options.fit_ratios:
        ref = values[reference] if abs(values[reference]) > 1e-10 else 1.0
        values = {name: (value/ref if name in visible else 0j) for name, value in values.items()}
        values[reference] = 1+0j
    parameter_names, initial, lower, upper = [], [], [], []
    def add(name, value, low=-np.inf, high=np.inf):
        parameter_names.append(name); initial.append(float(value)); lower.append(low); upper.append(high)
    if options.fit_orientation:
        add('orientation_deg', options.fixed_orientation_deg, -180, 360)
    add('log_amplitude', 0, -40, 40)
    if options.fit_background:
        for channel, y in zip(channels, ys):
            add('background_' + channel, np.percentile(y, 10))
    if options.fit_ratios:
        for name in visible:
            if name != reference:
                bound = options.ratio_bound
                add('ratio_' + name + '_real', np.clip(values[name].real, -0.99*bound, 0.99*bound), -bound, bound)
                if options.complex_ratios:
                    add('ratio_' + name + '_imag', np.clip(values[name].imag, -0.99*bound, 0.99*bound), -bound, bound)
    if sum(map(len, ys)) <= len(initial) + 2:
        raise ValueError('有效测量数量不足以支撑所选参数，请减少自由参数。')
    index = {name: i for i, name in enumerate(parameter_names)}
    def decode(parameters):
        orientation = parameters[index['orientation_deg']] if options.fit_orientation else options.fixed_orientation_deg
        coefficients = dict(values)
        if options.fit_ratios:
            for name in visible:
                if name != reference:
                    coefficients[name] = complex(parameters[index['ratio_' + name + '_real']], parameters[index['ratio_' + name + '_imag']] if options.complex_ratios else 0)
        return orientation, np.exp(parameters[index['log_amplitude']]), coefficients
    def model(parameters):
        orientation, amplitude, coefficients = decode(parameters)
        return [amplitude*predict(group, angle, orientation, coefficients, geometry, stack, channel) +
                (parameters[index['background_' + channel]] if options.fit_background else 0) for channel, angle in zip(channels, angles)]
    # Start amplitude from observed dynamic range, without normalizing channels separately.
    raw = model(np.asarray(initial))
    strength = max(np.max(v) - np.min(v) for v in ys)
    shape_strength = max(np.max(v) - np.min(v) for v in raw)
    if shape_strength < 1e-14:
        raise ValueError('当前张量参数给出常数或零图样；请调整独立参数或选择可辨识的模型。')
    initial[index['log_amplitude']] = np.clip(np.log(max(strength, 1e-12)/shape_strength), -35, 35)
    known_errors = all(s is not None for s in sigmas)
    if not known_errors:
        scale = max(np.std(np.concatenate(ys)), 1e-8)
        sigmas = [np.full(len(y), scale) for y in ys]
        warnings.append('没有完整的逐点误差：使用等权拟合，可信区间依赖残差噪声假设。')
    def check_cancel():
        if cancelled and cancelled():
            raise InterruptedError('已取消拟合。')
    def solve(targets, start):
        def residual(parameters):
            check_cancel()
            return np.concatenate([(prediction-y)/error for prediction, y, error in zip(model(parameters), targets, sigmas)])
        return least_squares(residual, start, bounds=(lower, upper), max_nfev=800, x_scale='jac')
    best, costs = None, []
    starts = np.linspace(0, orientation_period(group), options.starts, endpoint=False) if options.fit_orientation else [options.fixed_orientation_deg]*(options.starts if options.complex_ratios and options.fit_ratios else 1)
    for count, start_phi in enumerate(starts):
        p0 = np.array(initial)
        if options.fit_orientation:
            p0[index['orientation_deg']] = start_phi
        # All-real coefficients are a zero-gradient saddle for imaginary ratios.
        # Seed both phase signs rather than silently returning the real-only fit.
        for name, i in index.items():
            if name.startswith('ratio_') and name.endswith('_imag') and abs(p0[i]) < 1e-9:
                p0[i] = (1 if count % 2 == 0 else -1)*min(0.1, options.ratio_bound*0.1)
        result = solve(ys, p0)
        costs.append(float(result.cost))
        if result.success and (best is None or result.cost < best.cost):
            best = result
        if progress:
            progress(int(35*(count+1)/len(starts)))
    if best is None:
        raise ValueError('所有起点均未收敛，请检查数据与模型。')
    predicted = model(best.x)
    norms = np.linalg.norm(best.jac, axis=0)
    scaled_jac = best.jac/np.where(norms > 0, norms, 1)
    singular = np.linalg.svd(scaled_jac, compute_uv=False)
    rank = int(sum(singular > (singular[0]*1e-6 if len(singular) else 0)))
    identifiable = rank == len(initial)
    bounded = any(abs(best.x[i]) >= 0.995*options.ratio_bound for name, i in index.items() if name.startswith('ratio_')) or bool(best.active_mask[index['log_amplitude']])
    dof = sum(map(len, ys)) - len(initial)
    intervals = {}
    if identifiable and not bounded:
        inverse_norms = 1/np.where(norms > 0, norms, 1)
        covariance = np.linalg.pinv(scaled_jac.T @ scaled_jac)*inverse_norms[:, None]*inverse_norms[None, :]
        covariance *= 1 if known_errors else 2*best.cost/dof
        se = np.sqrt(np.maximum(np.diag(covariance), 0))
        intervals = {name: [float(v-1.96*e), float(v+1.96*e)] for name, v, e in zip(parameter_names, best.x, se)}
    elif not identifiable:
        warnings.append('雅可比矩阵秩不足：参数不能唯一辨识，已停止输出参数可信区间。请固定取向或减少张量比。')
    if bounded:
        warnings.append('参数触及设定边界：已停止输出可信区间。请调整参考分量或张量比上限后重新拟合。')
    period = orientation_period(group)
    bootstrap_values = []
    rng = np.random.default_rng(options.seed)
    if options.bootstrap and identifiable and not bounded:
        residuals = [(y-p)/s for y, p, s in zip(ys, predicted, sigmas)]
        residuals = [r-np.mean(r) for r in residuals]
        for attempt in range(options.bootstrap):
            check_cancel()
            targets = [p + s*rng.choice(r, size=len(r), replace=True) for p, s, r in zip(predicted, sigmas, residuals)]
            sample = solve(targets, best.x)
            if sample.success:
                point = sample.x.copy()
                if options.fit_orientation:
                    i = index['orientation_deg']
                    point[i] = best.x[i] + (point[i]-best.x[i]+period/2) % period-period/2
                bootstrap_values.append(point)
            if progress:
                progress(35 + int(65*(attempt+1)/options.bootstrap))
        if len(bootstrap_values) >= 20:
            low, high = np.percentile(bootstrap_values, [2.5, 97.5], axis=0)
            intervals = {name: [float(a), float(b)] for name, a, b in zip(parameter_names, low, high)}
        else:
            warnings.append('成功重采样少于 20 次，保留线性化可信区间。')
    orientation, amplitude, coefficients = decode(best.x)
    canonical = orientation % period
    if 'orientation_deg' in intervals:
        intervals['orientation_deg'] = [v-orientation+canonical for v in intervals['orientation_deg']]
    amplitude_interval = [float(np.exp(v)) for v in intervals.pop('log_amplitude')] if 'log_amplitude' in intervals else None
    intervals['amplitude'] = amplitude_interval
    parameters = {name: float(v) for name, v in zip(parameter_names, best.x) if name != 'log_amplitude'}
    parameters['amplitude'] = float(amplitude)
    if options.fit_orientation:
        parameters['orientation_deg'] = float(canonical)
    if not geometry.zero_calibrated:
        warnings.append('实验偏振零点未标定：取向仅相对于本次实验零点。')
    if not options.fit_ratios and len(visible) > 1:
        warnings.append('本次固定了手动张量比：取向和误差区间以这些固定值为条件。')
    if not geometry.gain_calibrated:
        warnings.append('两通道响应比未标定：结果以所填响应比为条件；张量比不能视作已标定材料常数。')
    if stack.mode == 'relative':
        warnings.append('未使用折射率修正：仅作相对图样分析，不输出绝对非线性系数。')
    if options.complex_ratios and options.fit_ratios:
        warnings.append('强度数据可能存在复张量相位的离散等价解，不能保证相位符号唯一。')
    if any(abs(v) >= 0.995*options.ratio_bound for name, v in parameters.items() if name.startswith('ratio_')):
        warnings.append('部分张量比接近边界，需检查模型和参数范围。')
    details = []
    rss = 0.0
    for channel, angle, y, prediction, error, mask in zip(channels, angles, ys, predicted, sigmas, masks):
        rss += float(np.sum((y-prediction)**2))
        for original_index, a, measured, fitted, sigma in zip(np.flatnonzero(mask), angle, y, prediction, error):
            details.append({'channel': channel, 'original_index': int(original_index), 'angle_deg': float(a), 'intensity': float(measured), 'prediction': float(fitted), 'residual': float(measured-fitted), 'uncertainty': float(sigma) if known_errors else None})
    return {'schema_version': 1, 'group': group, 'geometry': asdict(geometry), 'optical_stack': stack.to_dict(), 'options': asdict(options),
            'parameters': parameters, 'intervals_95': intervals, 'tensor_ratios': {k: [v.real, v.imag] for k, v in coefficients.items()},
            'reference_component': reference, 'observable_components': visible, 'unobservable_components': [p for p in labels if p not in visible],
            'orientation_period_deg': period, 'equivalent_orientation_deg': [float((canonical+k*period) % 360) for k in range(int(360/period))],
            'identifiable': identifiable, 'bounded_solution': bounded, 'jacobian_rank': rank, 'parameter_count': len(initial), 'dof': dof, 'rss': rss,
            'chi_squared': float(2*best.cost) if known_errors else None, 'weighted': known_errors,
            'bootstrap_successes': len(bootstrap_values), 'interval_method': 'stratified residual bootstrap' if len(bootstrap_values) >= 20 else 'local linearization' if identifiable and not bounded else 'unavailable',
            'multi_start_costs': costs, 'warnings': warnings, 'records': details,
            'model_scope': 'in-plane electric-dipole sheet; weak-sheet linear optical response neglected; isotropic passive substrate/film; plane-wave reciprocal Fresnel factors; fixed laboratory geometry; fitted relative intensity scale; no substrate nonlinear source, no high-NA vector field or absolute susceptibility calibration',
            'inputs': {name: deepcopy(curve.metadata) for name, curve in curves.items()}}


def make_demo():
    """Seeded synthetic D3h example; never used to infer a user's real material."""
    from point_groups import get_all_point_groups
    group = next(g for g in get_all_point_groups() if g.startswith('6̄m2 ='))
    geometry = Geometry(perpendicular_gain=1.2, gain_calibrated=True, zero_calibrated=True)
    rng = np.random.default_rng(20261026)
    angles = np.arange(0, 360, 12.0)
    curves = {}
    for channel, background in [('parallel', 3.0), ('perpendicular', 5.0)]:
        y = 1200*predict(group, angles, 17.3, {'yyy': 1}, geometry, channel=channel)+background+rng.normal(0, 12, len(angles))
        metadata = {'schema_version': 1, 'angle_kind': 'incident_polarization', 'angle_unit': 'degree', 'geometry': 'reflection', 'intensity_unit': 'input_units', 'channel': channel, 'sources': [], 'warnings': ['模拟示例，不是实测数据。'], 'synthetic': True, 'seed': 20261026,
                    'records': [{'uncertainty': 12.0, 'quality_flags': []} for _ in angles]}
        curves[channel] = ExperimentalCurve(angles.copy(), y, metadata)
    return curves, group, geometry
