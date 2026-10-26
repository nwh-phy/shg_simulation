import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from copy import deepcopy
import numpy as np
import pytest
from experimental_data import ExperimentalCurve
from joint_fit import make_demo, fit_joint, FitOptions
from reflection_model import Geometry, predict
from point_groups import get_all_point_groups, get_tensor_basis


def test_shared_orientation_recovery_and_reproducible_intervals():
    curves, group, geometry = make_demo()
    options = FitOptions(starts=4, bootstrap=20, seed=24)
    first = fit_joint(curves, group, geometry, options=options)
    second = fit_joint(curves, group, geometry, options=options)
    assert first['identifiable']
    assert first['parameters']['orientation_deg'] == pytest.approx(17.3, abs=0.12)
    assert first['parameters']['amplitude'] == pytest.approx(1200, rel=0.02)
    assert first['intervals_95'] == second['intervals_95']
    assert first['bootstrap_successes'] == 20
    assert len(first['records']) == 60
    assert first['orientation_period_deg'] == 60


def test_nonidentifiable_orientation_tensor_gauge_has_no_false_intervals():
    group = next(g for g in get_all_point_groups() if g.startswith('6̄ ='))
    values = dict(zip(get_tensor_basis(group), [1.0, 0.4]))
    angles = np.arange(0, 360, 12.0)
    curves = {channel: ExperimentalCurve(angles, 200*predict(group, angles, 12, values, channel=channel), {'angle_kind': 'incident_polarization', 'intensity_unit': 'input_units'}) for channel in ['parallel', 'perpendicular']}
    result = fit_joint(curves, group, options=FitOptions(fit_ratios=True, fit_background=False, starts=3, bootstrap=0), tensor_values=values)
    assert not result['identifiable']
    assert result['interval_method'] == 'unavailable'
    assert result['intervals_95'].get('orientation_deg') is None


def test_explicit_exclusions_saturation_and_quantity_checks():
    curves, group, geometry = make_demo()
    curves['parallel'].metadata['records'][0]['quality_flags'] = ['saturated']
    with pytest.raises(ValueError, match='饱和'):
        fit_joint(curves, group, geometry, options=FitOptions(bootstrap=0))
    curves['parallel'].metadata['excluded_indices'] = [0]
    result = fit_joint(curves, group, geometry, options=FitOptions(starts=3, bootstrap=0))
    assert len(result['records']) == 59
    assert not any(r['channel'] == 'parallel' and r['original_index'] == 0 for r in result['records'])
    curves['parallel'].metadata['extraction'] = {'method': 'integral'}
    curves['perpendicular'].metadata['extraction'] = {'method': 'nearest'}
    with pytest.raises(ValueError, match='物理量不一致'):
        fit_joint(curves, group, geometry)


def test_fit_cancellation_and_missing_channel():
    curves, group, geometry = make_demo()
    before = deepcopy(curves['parallel'].intensity)
    with pytest.raises(InterruptedError):
        fit_joint(curves, group, geometry, cancelled=lambda: True)
    np.testing.assert_array_equal(curves['parallel'].intensity, before)
    with pytest.raises(ValueError, match='两组数据'):
        fit_joint({'parallel': curves['parallel']}, group)


def test_complex_ratio_recovery_from_real_initial_values():
    group = next(g for g in get_all_point_groups() if g.startswith('6̄ ='))
    names = list(get_tensor_basis(group))
    coefficients = {names[0]: 1, names[1]: 0.3+0.5j}
    angles = np.arange(0, 360, 12.0)
    curves = {c: ExperimentalCurve(angles, 150*predict(group, angles, 0, coefficients, channel=c), {'intensity_unit': 'input_units'}) for c in ['parallel', 'perpendicular']}
    result = fit_joint(curves, group, options=FitOptions(fit_ratios=True, complex_ratios=True, fit_background=False, fit_orientation=False, starts=4, bootstrap=0))
    assert result['identifiable']
    assert result['parameters']['ratio_'+names[1]+'_real'] == pytest.approx(0.3, abs=1e-4)
    assert abs(result['parameters']['ratio_'+names[1]+'_imag']) == pytest.approx(0.5, abs=1e-4)


def test_boundary_suppresses_intervals_and_reference_can_be_changed():
    group = next(g for g in get_all_point_groups() if g.startswith('6̄ ='))
    names = list(get_tensor_basis(group))
    coefficients = {names[0]: 1, names[1]: 3}
    angles = np.arange(0, 360, 12.0)
    curves = {c: ExperimentalCurve(angles, 200*predict(group, angles, 0, coefficients, channel=c), {'intensity_unit': 'input_units'}) for c in ['parallel', 'perpendicular']}
    bounded = fit_joint(curves, group, options=FitOptions(fit_ratios=True, fit_background=False, fit_orientation=False, bootstrap=0, ratio_bound=0.5))
    assert bounded['bounded_solution'] and bounded['interval_method'] == 'unavailable'
    assert all(value is None for value in bounded['intervals_95'].values())
    unbounded = fit_joint(curves, group, options=FitOptions(fit_ratios=True, fit_background=False, fit_orientation=False, bootstrap=0, ratio_bound=0.5, reference_component=names[1]))
    assert not unbounded['bounded_solution'] and unbounded['rss'] < 1e-12
