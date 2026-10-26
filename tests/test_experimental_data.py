import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.experimental_data import (
    ExtractionSettings, load_angle_curve, load_spectral_table,
    load_spectrum_folder, extract_curve, export_curve,
)


def write(path, content):
    path.write_text(content, encoding="utf-8-sig")
    return path


def spectrum_file(directory, sequence, content=None, suffix=""):
    return write(directory / f"sample--{sequence:03d}--Spec.Data 1{suffix}.txt", content or "529 5\n530 5\n531.9 12\n532.3 14\n534 7\n535 7\n")


@pytest.mark.parametrize("header", ["", "angle_deg,intensity\n", "Angle,Intensity\n"])
def test_curve_preserves_first_row_repeats_negative_and_unwrapped_angles(tmp_path, header):
    path = write(tmp_path / "curve.csv", header + "0,2\n0,3\n360,-1\n372,4\n")
    curve = load_angle_curve(path)
    np.testing.assert_array_equal(curve.angle_deg, [0, 0, 360, 372])
    np.testing.assert_array_equal(curve.intensity, [2, 3, -1, 4])
    assert curve.metadata["angle_kind"] == "incident_polarization"
    assert curve.metadata["warnings"]


@pytest.mark.parametrize("contents", ["0,NaN\n", "0,1\n12,bad\n", "0,,2\n", "Wavelength,Intensity\n532,7\n", "0,intensity\n12,4\n"])
def test_bad_curve_is_rejected_without_skipping_rows(tmp_path, contents):
    with pytest.raises(ValueError):
        load_angle_curve(write(tmp_path / "bad.csv", contents))


def test_long_table_groups_and_sorts_each_wavelength_axis(tmp_path):
    path = write(tmp_path / "raw.csv", "angle_deg,wavelength_nm,intensity\n12,533,30\n0,531,10\n12,531,20\n0,533,14\n")
    spectra, metadata = load_spectral_table(path)
    assert [s.angle_deg for s in spectra] == [12, 0]
    np.testing.assert_array_equal(spectra[0].wavelength_nm, [531, 533])
    curve = extract_curve(spectra, ExtractionSettings(532, method="linear", max_distance_nm=1), metadata)
    np.testing.assert_allclose(curve.intensity, [25, 12])
    with pytest.raises(ValueError, match="重复波长"):
        load_spectral_table(write(tmp_path / "repeated.csv", "0 532 2\n0 532 3\n"))


def test_folder_step_is_explicit_and_endpoints_are_not_fabricated(tmp_path):
    for sequence in [0, 1, 30, 31]:
        spectrum_file(tmp_path, sequence)
    with pytest.raises(TypeError):
        load_spectrum_folder(tmp_path)
    spectra, metadata = load_spectrum_folder(tmp_path, step_deg=12)
    assert [s.angle_deg for s in spectra] == [0, 12, 360, 372]
    assert metadata["angle_mapping"]["factor_to_polarization"] == 1
    assert metadata["sequence_gaps"] == [[2, 29]]
    negative, _ = load_spectrum_folder(tmp_path, step_deg=-7.5, start_deg=90)
    assert [s.angle_deg for s in negative] == [90, 82.5, -135, -142.5]
    with pytest.raises(ValueError):
        load_spectrum_folder(tmp_path, step_deg=0)


def test_exact_copies_logged_and_different_repeats_preserved(tmp_path):
    first = spectrum_file(tmp_path, 0)
    spectrum_file(tmp_path, 0, suffix=" (1)")
    spectrum_file(tmp_path, 0, content="531 4\n532 9\n533 6\n", suffix=" (2)")
    spectra, metadata = load_spectrum_folder(tmp_path, step_deg=12)
    assert len(spectra) == 2
    assert len(metadata["excluded_identical_copies"]) == 1
    assert metadata["warnings"]
    assert metadata["sources"][0]["sha256"]
    assert first.read_text(encoding="utf-8-sig").startswith("529 5")


@pytest.mark.parametrize("method,expected,selected", [("nearest", 12, 531.9), ("linear", 12.5, 532), ("window_max", 14, 532.3)])
def test_extraction_methods_are_distinct_and_unprocessed_by_default(tmp_path, method, expected, selected):
    path = spectrum_file(tmp_path, 0)
    before = path.read_bytes()
    spectra, metadata = load_spectrum_folder(tmp_path, 12)
    curve = extract_curve(spectra, ExtractionSettings(532, method=method), metadata)
    np.testing.assert_allclose(curve.intensity, [expected])
    assert curve.metadata["records"][0]["sample_wavelength_nm"] == selected
    assert curve.metadata["records"][0]["background"] == 0
    assert "extraction" not in metadata
    assert path.read_bytes() == before


def test_explicit_background_and_negative_signal(tmp_path):
    spectrum_file(tmp_path, 0)
    spectra, metadata = load_spectrum_folder(tmp_path, 12)
    settings = ExtractionSettings(532, background_ranges_nm=((529, 530), (534, 535)))
    curve = extract_curve(spectra, settings, metadata)
    np.testing.assert_allclose(curve.intensity, [6])
    assert curve.metadata["records"][0]["background"] == 6
    negative = ExtractionSettings(532, background_ranges_nm=((532.2, 532.4),))
    np.testing.assert_allclose(extract_curve(spectra, negative, metadata).intensity, [-2])


def test_invalid_windows_and_target_fail_without_mutating_metadata(tmp_path):
    spectrum_file(tmp_path, 0)
    spectra, metadata = load_spectrum_folder(tmp_path, 12)
    before = json.dumps(metadata, sort_keys=True)
    for settings in [ExtractionSettings(800), ExtractionSettings(532, max_distance_nm=0.01), ExtractionSettings(532, background_ranges_nm=((520, 525),)), ExtractionSettings(532, background_ranges_nm=((530.1, 530.2),))]:
        with pytest.raises(ValueError):
            extract_curve(spectra, settings, metadata)
    assert json.dumps(metadata, sort_keys=True) == before
    with pytest.raises(ValueError, match="背景区间"):
        ExtractionSettings(532, background_ranges_nm=((531, 533),))
    with pytest.raises(ValueError, match="重叠"):
        ExtractionSettings(532, method="window_max", background_ranges_nm=((530, 531.5),))


def test_export_roundtrip_provenance_and_no_overwrite(tmp_path):
    source = write(tmp_path / "curve.txt", "0 7\n12 9\n")
    curve = load_angle_curve(source)
    curve.metadata["channel"] = "parallel"
    csv_path, metadata_path = export_curve(curve, tmp_path / "new_export")
    loaded = load_angle_curve(csv_path)
    np.testing.assert_array_equal(loaded.intensity, curve.intensity)
    assert loaded.metadata["channel"] == "parallel"
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["sources"][0]["path"] == str(source.resolve())
    with pytest.raises(FileExistsError):
        export_curve(curve, tmp_path / "new_export")
    with csv_path.open("a", encoding="utf-8") as stream:
        stream.write("24,10\n")
    with pytest.raises(ValueError, match="校验值"):
        load_angle_curve(csv_path)


def test_cli_refuses_missing_step_and_creates_no_output(tmp_path):
    entry = Path(__file__).resolve().parents[1] / "src" / "import_shg_data.py"
    result = subprocess.run([sys.executable, "-X", "utf8", "-B", str(entry), str(tmp_path), "--kind", "folder", "--target-nm", "532", "--output", str(tmp_path / "export")], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 2
    assert "--step-deg" in result.stderr
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize("replacement", [[], {"schema_version": 2}, {"schema_version": 1, "warnings": "invalid"}])
def test_invalid_metadata_fails_as_a_visible_import_error(tmp_path, replacement):
    curve = load_angle_curve(write(tmp_path / "source.csv", "0,1\n0,2\n"))
    csv_path, metadata_path = export_curve(curve, tmp_path / "bundle")
    metadata_path.write_text(json.dumps(replacement), encoding="utf-8")
    with pytest.raises(ValueError, match="元数据"):
        load_angle_curve(csv_path)
