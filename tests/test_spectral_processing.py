import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
from scipy.special import erf
from experimental_data import Spectrum, ExtractionSettings, extract_curve


def metadata():
    return {'schema_version': 1, 'angle_kind': 'incident_polarization', 'warnings': [], 'sources': []}


def test_linear_baseline_gaussian_area_and_exposure_power_scaling():
    x = np.linspace(528, 536, 801)
    y = 10 + 2*(x-532) + 100*np.exp(-0.5*((x-532)/0.2)**2)
    spectrum = Spectrum(12, x, y, 'synthetic Gaussian')
    settings = ExtractionSettings(532, 'integral', 1, 0.5, ((528, 530), (534, 536)), 'linear', 2, 3)
    curve = extract_curve([spectrum], settings, metadata())
    expected = 100*0.2*np.sqrt(2*np.pi)*erf(1/(np.sqrt(2)*0.2))/18
    assert curve.intensity[0] == pytest.approx(expected, rel=1e-6)
    assert curve.metadata['intensity_unit'] == 'input_units*nm/s/mW^2'
    assert curve.metadata['records'][0]['baseline_coefficients'] == pytest.approx([10, 2])


def test_background_uncertainty_and_saturation_review_never_delete_points():
    x = np.arange(528, 537, 0.1)
    rng = np.random.default_rng(42)
    y = 100 + rng.normal(0, 2, len(x))
    y[np.argmin(abs(x-532))] = 1000
    settings = ExtractionSettings(532, background_ranges_nm=((528, 530), (534, 536)), saturation_level=900)
    curve = extract_curve([Spectrum(0, x, y, 'synthetic noisy detector')], settings, metadata())
    record = curve.metadata['records'][0]
    assert record['uncertainty'] > 0
    assert 'saturated' in record['quality_flags']
    assert 'isolated_spike_review' in record['quality_flags']
    assert len(curve.intensity) == 1 and 'excluded_indices' not in curve.metadata


def test_integration_does_not_bridge_missing_spectral_samples():
    x = np.array([529, 530, 531, 533, 534, 535], float)
    with pytest.raises(ValueError, match='缺口'):
        extract_curve([Spectrum(0, x, np.ones(len(x)), 'gap')], ExtractionSettings(532, 'integral', 2), metadata())
