"""SHG experimental imports. Angles are already converted incident polarization angles."""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv
import hashlib
import io
import json
import re
import sys

import numpy as np


@dataclass(frozen=True)
class ExtractionSettings:
    target_nm: float
    method: str = "nearest"
    half_width_nm: float = 1.0
    max_distance_nm: float = 0.5
    background_ranges_nm: Tuple[Tuple[float, float], ...] = ()
    background_model: str = 'constant'
    exposure_s: Optional[float] = None
    power_mw: Optional[float] = None
    saturation_level: Optional[float] = None
    spike_sigma: float = 8.0

    def __post_init__(self):
        values = (self.target_nm, self.half_width_nm, self.max_distance_nm)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("目标波长、窗口半宽和允许偏差必须为有限正数。")
        if self.method not in ("nearest", "linear", "window_max", "integral"):
            raise ValueError("未知提取方法。")
        if self.background_model not in ('constant', 'linear'):
            raise ValueError('背景模型必须为常数或线性。')
        for value in (self.exposure_s, self.power_mw, self.saturation_level, self.spike_sigma):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError('曝光、功率、饱和阈值和异常像素阈值必须为有限正数。')
        for low, high in self.background_ranges_nm:
            if not np.isfinite([low, high]).all() or not 0 < low < high:
                raise ValueError("背景区间必须满足 0 < 下限 < 上限。")
            signal_low = self.target_nm - (self.half_width_nm if self.method in ("window_max", "integral") else 0)
            signal_high = self.target_nm + (self.half_width_nm if self.method in ("window_max", "integral") else 0)
            if low <= signal_high and high >= signal_low:
                raise ValueError("背景区间不能包含目标波长或与信号窗口重叠。")


@dataclass
class Spectrum:
    angle_deg: float
    wavelength_nm: np.ndarray
    intensity: np.ndarray
    source: str


@dataclass
class ExperimentalCurve:
    angle_deg: np.ndarray
    intensity: np.ndarray
    metadata: Dict
    spectra: List[Spectrum] = field(default_factory=list)


def _fingerprint(path: Path) -> Dict:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _read_table(path: Path, columns: int) -> Tuple[np.ndarray, Optional[List[str]]]:
    """Read complete numeric rows; never discard the first measured row or bad rows."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding="gb18030")
    rows, header = [], None
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        separator = "," if "," in line else ";" if ";" in line else None
        cells = next(csv.reader([line], delimiter=separator)) if separator else line.split()
        if len(cells) != columns:
            raise ValueError(f"{path.name} 第 {number} 行应有 {columns} 列，实际为 {len(cells)} 列。")
        try:
            values = [float(c) for c in cells]
        except ValueError:
            labels = [c.strip().lower() for c in cells]
            angle_labels = {"angle", "angle_deg", "alpha", "alpha_deg", "角度", "角度_deg"}
            wave_labels = {"wavelength", "wavelength_nm", "wavelength(nm)", "lambda_nm", "波长", "波长_nm", "波长(nm)"}
            intensity_labels = {"intensity", "counts", "count", "强度"}
            valid_header = (labels[0] in (angle_labels | wave_labels) and labels[1] in intensity_labels) if columns == 2 else (labels[0] in angle_labels and labels[1] in wave_labels and labels[2] in intensity_labels)
            if not rows and header is None and valid_header:
                header = [c.strip() for c in cells]
                continue
            raise ValueError(f"{path.name} 第 {number} 行包含无效数值；未跳过该行。") from None
        if not np.isfinite(values).all():
            raise ValueError(f"{path.name} 第 {number} 行含 NaN 或无穷值。")
        rows.append(values)
    if not rows:
        raise ValueError(f"{path.name} 没有有效数据。")
    return np.asarray(rows, dtype=float), header


def _metadata(mode: str, sources: List[Dict]) -> Dict:
    return {
        "schema_version": 1,
        "angle_kind": "incident_polarization",
        "angle_unit": "degree",
        "geometry": "reflection",
        "channel": "unspecified",
        "intensity_unit": "input_units",
        "input_mode": mode,
        "sources": sources,
        "warnings": [],
    }


def load_angle_curve(path) -> ExperimentalCurve:
    """Two columns: angle_deg, intensity. Preserve repeats, endpoints and negative values."""
    path = Path(path)
    data, header = _read_table(path, 2)
    if header and any(s in header[0].lower() for s in ("wave", "lambda", "波长")):
        raise ValueError("此文件表头为波长—强度，请使用逐文件光谱目录入口。")
    metadata = _metadata("angle_intensity", [_fingerprint(path)])
    sidecar = path.with_suffix(".metadata.json")
    if sidecar.exists():
        previous = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(previous, dict) or previous.get("schema_version") != 1:
            raise ValueError("同名元数据格式或版本无效，应为 SHG 实验数据格式 v1。")
        if not isinstance(previous.get("warnings"), list) or not all(isinstance(item, str) for item in previous["warnings"]):
            raise ValueError("同名元数据中的 warnings 字段格式无效。")
        if previous.get("export_csv_sha256") != metadata["sources"][0]["sha256"]:
            raise ValueError("数据与同名 metadata.json 的校验值不一致，请核实文件来源。")
        if previous.get("angle_kind") != "incident_polarization":
            raise ValueError("元数据未将角度标为已换算的入射偏振角。")
        metadata = previous
        metadata["reimported_from"] = _fingerprint(path)
    metadata["row_count"] = len(data)
    repeat_message = "包含重复角度，已逐条保留，未平均或删除。"
    if len(np.unique(data[:, 0])) < len(data) and repeat_message not in metadata["warnings"]:
        metadata["warnings"].append(repeat_message)
    return ExperimentalCurve(data[:, 0], data[:, 1], metadata)


def _spectrum(angle: float, data: np.ndarray, source: str) -> Spectrum:
    order = np.argsort(data[:, 0], kind="stable")
    x, y = data[order, 0], data[order, 1]
    if len(x) < 2 or np.any(x <= 0) or np.any(np.diff(x) <= 0):
        raise ValueError(f"{source}：每条光谱需至少两个不同的正波长；重复波长未自动合并。")
    return Spectrum(float(angle), x.copy(), y.copy(), source)


def load_spectral_table(path) -> Tuple[List[Spectrum], Dict]:
    """Long table: angle_deg, wavelength_nm, intensity (wavelength in nm)."""
    path = Path(path)
    data, _ = _read_table(path, 3)
    # dict preserves acquisition order rather than folding angles modulo 360.
    angles = dict.fromkeys(data[:, 0])
    spectra = [_spectrum(a, data[data[:, 0] == a, 1:], str(path.resolve())) for a in angles]
    return spectra, _metadata("angle_wavelength_intensity", [_fingerprint(path)])


def load_spectrum_folder(directory, step_deg: float, start_deg: float = 0.0) -> Tuple[List[Spectrum], Dict]:
    """Sequence --NNN-- maps to start + NNN * step; no wrapping or angle doubling."""
    directory = Path(directory)
    if not np.isfinite([step_deg, start_deg]).all() or step_deg == 0:
        raise ValueError("起始角必须有限，角度步长必须有限且非零。")
    paths = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in (".txt", ".csv", ".tsv") and not p.name.startswith("~$"))
    if not paths:
        raise ValueError("所选目录没有 TXT、CSV 或 TSV 光谱文件。")
    spectra, sources, excluded, seen = [], [], [], {}
    for path in paths:
        match = re.search(r"--(\d+)--", path.name)
        if not match:
            raise ValueError(f"无法从 {path.name} 读取 --序号--，请单独选择一个测量序列目录。")
        sequence = int(match.group(1))
        source = _fingerprint(path)
        identity = (sequence, source["sha256"])
        if identity in seen:
            excluded.append({**source, "reason": "identical_copy_same_sequence", "identical_to": seen[identity]})
            continue
        data, header = _read_table(path, 2)
        if header and any(s in header[0].lower() for s in ("angle", "角度")):
            raise ValueError(f"{path.name} 的表头是角度，请选择波长—强度光谱目录。")
        angle = start_deg + sequence * step_deg
        spectra.append(_spectrum(angle, data, str(path.resolve())))
        sources.append({**source, "sequence": sequence, "angle_deg": angle})
        seen[identity] = str(path.resolve())
    metadata = _metadata("spectrum_folder", sources)
    metadata["angle_mapping"] = {"start_deg": start_deg, "step_deg": step_deg, "formula": "start_deg + sequence * step_deg", "factor_to_polarization": 1}
    metadata["excluded_identical_copies"] = excluded
    sequence_ids = sorted({s["sequence"] for s in sources})
    metadata["sequence_gaps"] = [[a + 1, b - 1] for a, b in zip(sequence_ids, sequence_ids[1:]) if b > a + 1]
    if metadata["sequence_gaps"]:
        metadata["warnings"].append("文件序号有缺口，已按原编号映射角度，未补测量点。")
    if sequence_ids[0] != 0:
        metadata["warnings"].append("最早序号不是 000；起始角仍指序号 000，请核对采集约定。")
    if excluded:
        metadata["warnings"].append(f"有 {len(excluded)} 个同序号且字节相同的副本，已在元数据中列出。")
    if len({s.angle_deg for s in spectra}) < len(spectra):
        metadata["warnings"].append("同序号的不同光谱作为重复测量逐条保留。")
    return spectra, metadata


def _linear_weights(x, target, tolerance):
    weights = np.zeros(len(x))
    index = int(np.searchsorted(x, target))
    if index < len(x) and x[index] == target:
        weights[index] = 1
    else:
        if index == 0 or index == len(x):
            raise ValueError('目标波长超出光谱范围。')
        if max(target-x[index-1], x[index]-target) > tolerance:
            raise ValueError('插值所需采样点超出允许波长偏差，请检查采样间隔或单位。')
        fraction = (target-x[index-1])/(x[index]-x[index-1])
        weights[index-1], weights[index] = 1-fraction, fraction
    return weights


def extract_curve(spectra: List[Spectrum], settings: ExtractionSettings, metadata: Dict) -> ExperimentalCurve:
    """Explicit spectral processing; linear uncertainty propagation uses background iid noise."""
    if not spectra:
        raise ValueError('没有可提取的光谱。')
    records = []
    for spectrum in spectra:
        x, y, target = spectrum.wavelength_nm, spectrum.intensity, settings.target_nm
        if not x[0] <= target <= x[-1]:
            raise ValueError(f'{spectrum.source}: 目标 {target:g} nm 超出光谱范围。')
        baseline_mask = np.zeros(len(x), bool)
        for low, high in settings.background_ranges_nm:
            selected = (x >= low) & (x <= high)
            if low < x[0] or high > x[-1] or not selected.any():
                raise ValueError(f'{spectrum.source}: 背景波段未被光谱完整覆盖或没有采样点。')
            baseline_mask |= selected
        design = np.column_stack([np.ones(len(x)), x-target]) if settings.background_model == 'linear' else np.ones((len(x), 1))
        beta = np.zeros(design.shape[1])
        noise_variance = None
        background_covariance = None
        if baseline_mask.any():
            X = design[baseline_mask]
            if len(X) < X.shape[1] or np.linalg.matrix_rank(X) != X.shape[1]:
                raise ValueError('背景采样点不足以支持所选背景模型。')
            beta = np.linalg.lstsq(X, y[baseline_mask], rcond=None)[0]
            dof = len(X)-X.shape[1]
            if dof > 0:
                noise_variance = float(np.sum((y[baseline_mask]-X@beta)**2)/dof)
                floor = np.finfo(float).eps * max(float(np.mean(y[baseline_mask]**2)), 1.0)
                if noise_variance <= floor:
                    noise_variance = None
                else:
                    background_covariance = noise_variance*np.linalg.pinv(X.T@X)
        weights = np.zeros(len(x))
        if settings.method == 'nearest':
            index = int(np.argmin(np.abs(x-target)))
            if abs(x[index]-target) > settings.max_distance_nm:
                raise ValueError(f'{spectrum.source}: 最近实测点超出允许波长偏差。')
            weights[index] = 1
            sample_nm = float(x[index])
        elif settings.method == 'linear':
            weights = _linear_weights(x, target, settings.max_distance_nm)
            sample_nm = float(target)
        elif settings.method == 'window_max':
            low, high = target-settings.half_width_nm, target+settings.half_width_nm
            if low < x[0] or high > x[-1]:
                raise ValueError('光谱未完整覆盖信号窗口。')
            indices = np.flatnonzero((x >= low) & (x <= high))
            if not len(indices):
                raise ValueError('信号窗口内没有实测点。')
            # Select the maximum AFTER the specified background correction.
            index = int(indices[np.argmax((y-design@beta)[indices])])
            weights[index] = 1
            sample_nm = float(x[index])
        else:
            low, high = target-settings.half_width_nm, target+settings.half_width_nm
            if low < x[0] or high > x[-1]:
                raise ValueError('光谱未完整覆盖积分窗口。')
            knots = np.r_[low, x[(x > low) & (x < high)], high]
            if np.max(np.diff(knots)) > 2*settings.max_distance_nm:
                raise ValueError('积分窗口包含超出允许采样间距的波长缺口。')
            coefficients = np.r_[np.diff(knots)/2, 0] + np.r_[0, np.diff(knots)/2]
            for knot, coefficient in zip(knots, coefficients):
                weights += coefficient*_linear_weights(x, knot, settings.max_distance_nm)
            sample_nm = float(target)
        used = weights != 0
        if (used & baseline_mask).any():
            raise ValueError('信号采样点与背景采样点重叠。')
        raw = float(weights@y)
        background_vector = weights@design
        background = float(background_vector@beta)
        normalization = (settings.exposure_s or 1)*(settings.power_mw or 1)**2
        uncertainty = None
        if noise_variance is not None and noise_variance > 0 and settings.method != 'window_max':
            variance = noise_variance*float(weights@weights) + float(background_vector@background_covariance@background_vector)
            uncertainty = float(np.sqrt(max(variance, 0))/normalization)
        flags = []
        if settings.saturation_level is not None and np.any(y[used] >= settings.saturation_level):
            flags.append('saturated')
        if noise_variance is not None and noise_variance > 0:
            threshold = settings.spike_sigma*np.sqrt(noise_variance)
            for i in np.flatnonzero(used):
                if 0 < i < len(y)-1 and y[i]-max(y[i-1], y[i+1]) > threshold:
                    flags.append('isolated_spike_review')
                    break
        records.append({'angle_deg': spectrum.angle_deg, 'intensity': (raw-background)/normalization,
                        'sample_wavelength_nm': sample_nm, 'raw_intensity': raw, 'background': background,
                        'baseline_coefficients': beta.tolist(), 'preview_raw_intensity': float(np.interp(sample_nm, x, y)),
                        'uncertainty': uncertainty, 'quality_flags': flags,
                        'uncertainty_model': 'background iid residual noise + background-fit covariance; not calibrated detector noise' if uncertainty is not None else 'not estimated; peak selection uncertainty not propagated for window_max',
                        'source': spectrum.source})
    result_metadata = json.loads(json.dumps(metadata, ensure_ascii=False))
    result_metadata['extraction'] = asdict(settings)
    result_metadata['records'] = records
    result_metadata['row_count'] = len(records)
    unit = 'input_units*nm' if settings.method == 'integral' else 'input_units'
    if settings.exposure_s is not None:
        unit += '/s'
    if settings.power_mw is not None:
        unit += '/mW^2'
    result_metadata['intensity_unit'] = unit
    result_metadata['uncertainty_scope'] = 'Conditional iid background-noise propagation, not including wavelength calibration, baseline model selection, laser drift or peak selection.'
    if any(r['quality_flags'] for r in records):
        result_metadata['warnings'].append('部分光谱有质量提示，请查看原谱并人工复核；未自动删点或做角度插值。')
    return ExperimentalCurve(np.array([r['angle_deg'] for r in records]), np.array([r['intensity'] for r in records]), result_metadata, spectra)


def export_curve(curve: ExperimentalCurve, directory) -> Tuple[Path, Path]:
    """Create a new export directory. Existing outputs and raw files are never overwritten."""
    directory = Path(directory)
    if np.ndim(curve.angle_deg) != 1 or np.ndim(curve.intensity) != 1 or len(curve.angle_deg) != len(curve.intensity) or not len(curve.angle_deg):
        raise ValueError("角度和强度数量不匹配或为空。")
    if not np.isfinite(curve.angle_deg).all() or not np.isfinite(curve.intensity).all():
        raise ValueError("不能导出非有限数值。")
    # Validate and serialize both artifacts before creating any output directory.
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["angle_deg", "intensity"])
    writer.writerows(zip(curve.angle_deg, curve.intensity))
    csv_text = buffer.getvalue()
    metadata = dict(curve.metadata)
    metadata["exported_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["export_csv_sha256"] = hashlib.sha256(csv_text.encode("utf-8")).hexdigest()
    metadata["software"] = {"python": sys.version.split()[0], "numpy": np.__version__, "format": "shg_experimental_v1"}
    metadata_text = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)
    directory.mkdir(parents=True, exist_ok=False)
    csv_path = directory / "shg_curve.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(csv_text)
    metadata_path = csv_path.with_suffix(".metadata.json")
    with metadata_path.open("x", encoding="utf-8") as stream:
        stream.write(metadata_text)
    return csv_path, metadata_path
