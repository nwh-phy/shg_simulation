"""Validated projects and fresh, traceable analysis export bundles."""
from datetime import datetime, timezone
from pathlib import Path
import csv
import html
import json
import os
import tempfile
import numpy as np
from experimental_data import ExperimentalCurve
from joint_fit import FitOptions
from reflection_model import Geometry, OpticalStack
from point_groups import get_all_point_groups, get_components_for_group

APP_VERSION = '2.0.0'


def parameter_label(name):
    labels = {'orientation_deg': '相对实验零点取向（°）', 'amplitude': '共享强度比例 A',
              'background_parallel': '平行通道偏置 B∥', 'background_perpendicular': '垂直通道偏置 B⊥'}
    if name.startswith('ratio_'):
        return '张量比 ' + name[6:].replace('_real', '（实部）').replace('_imag', '（虚部）')
    return labels.get(name, name)


def curve_dict(curve):
    return {'angle_deg': curve.angle_deg.tolist(), 'intensity': curve.intensity.tolist(), 'metadata': curve.metadata}


def validate_project(document):
    if not isinstance(document, dict) or document.get('project_schema') != 1:
        raise ValueError('不是支持的 SHG 项目文件。')
    geometry = Geometry(**document['geometry'])
    optics = OpticalStack.from_dict(document['optical_stack'])
    options = FitOptions(**document['fit_options'])
    group = document['group']
    if group and group not in get_all_point_groups():
        raise ValueError('项目中的点群不存在。')
    curves = {}
    for name, value in document['curves'].items():
        if name not in ('parallel', 'perpendicular'):
            raise ValueError('项目含未知通道。')
        angle, intensity = np.asarray(value['angle_deg'], float), np.asarray(value['intensity'], float)
        if angle.ndim != 1 or intensity.shape != angle.shape or not len(angle) or not np.isfinite(angle).all() or not np.isfinite(intensity).all():
            raise ValueError('项目的角度或强度无效。')
        metadata = value['metadata']
        if not isinstance(metadata, dict) or metadata.get('angle_kind') != 'incident_polarization':
            raise ValueError('项目未声明角度为已换算的入射偏振角。')
        curves[name] = ExperimentalCurve(angle, intensity, metadata)
    values = {name: complex(v) for name, v in document.get('tensor_values', {}).items()}
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError('项目张量参数无效。')
    if group and set(values) - set(get_components_for_group(group)):
        raise ValueError('项目包含不属于此点群的独立分量。')
    return curves, group, geometry, optics, options, values


def save_project(path, document, overwrite=False):
    validate_project(document)
    text = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError('项目文件已存在。')
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix='shg_project_', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_project(path):
    document = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    return document, validate_project(document)


def export_analysis(directory, result, figure, project):
    validate_project(project)
    result_text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    project_text = json.dumps(project, ensure_ascii=False, indent=2, allow_nan=False)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'fit_result.json').write_text(result_text, encoding='utf-8')
    (directory/'project.shg.json').write_text(project_text, encoding='utf-8')
    with (directory/'measurements_and_residuals.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['channel', 'original_index', 'angle_deg', 'intensity', 'prediction', 'residual', 'uncertainty'])
        writer.writeheader(); writer.writerows(result['records'])
    with (directory/'parameters.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['parameter', 'value', 'ci95_low', 'ci95_high'])
        for name, value in result['parameters'].items():
            bounds = result['intervals_95'].get(name)
            writer.writerow([name, value, *(bounds or [None, None])])
    figure.savefig(directory/'fit.png', dpi=180)
    figure.savefig(directory/'fit.pdf')
    rows = []
    for name, value in result['parameters'].items():
        bounds = result['intervals_95'].get(name)
        interval = f'{bounds[0]:.6g} – {bounds[1]:.6g}' if bounds else '不可辨识或未估计'
        rows.append(f'<tr><td>{html.escape(parameter_label(name))}</td><td>{value:.7g}</td><td>{interval}</td></tr>')
    warnings = ''.join(f'<li>{html.escape(w)}</li>' for w in result['warnings'])
    synthetic = any(v.get('synthetic') for v in result['inputs'].values())
    label = '模拟数据演示' if synthetic else '实验数据拟合'
    report = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SHG 分析报告</title>
<style>body{{font:16px/1.6 "Microsoft YaHei",sans-serif;max-width:1100px;margin:32px auto;color:#18202a}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #bbb;padding:8px}}img{{width:100%;height:auto}}.note{{background:#fff2d3;padding:16px}}</style>
<h1>SHG {label}</h1><p>软件 {APP_VERSION}；生成于 {datetime.now(timezone.utc).isoformat()}（UTC）。</p>
<p>点群：{html.escape(result['group'])}；角度为已换算的入射偏振角。拟合采用两个通道的原始相对强度，未分别归一化。</p>
<div class="note"><ul>{warnings}</ul></div><img src="fit.png" alt="数据、联合拟合与残差">
<table><tr><th>参数</th><th>估计</th><th>条件性 95% 区间</th></tr>{''.join(rows)}</table>
<p>取向等价周期 {result['orientation_period_deg']:g}°；跨周期区间按展开角理解。方法：{html.escape(result['interval_method'])}；参数秩 {result['jacobian_rank']}/{result['parameter_count']}，自由度 {result['dof']}。</p>
<p>可观测分量：{html.escape(', '.join(result['observable_components']))}。不可观测分量不作为测量结果：{html.escape(', '.join(result['unobservable_components']) or '无')}。</p>
<p>模型范围：{html.escape(result['model_scope'])}</p><p>几何、光学常数、固定张量比、随机种子、光谱处理记录和输入文件哈希见 <a href="fit_result.json">完整结果</a>；可通过软件打开 <a href="project.shg.json">项目文件</a>继续分析。</p>
<p>参考：<a href="https://arxiv.org/abs/1304.4289">二维材料偏振 SHG</a>；<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC4078302/">片层与基底局域场</a>；<a href="https://doi.org/10.1038/s41524-022-00930-4">反射 SHG 的边界条件</a>。</p></html>'''
    (directory/'report.html').write_text(report, encoding='utf-8')
    return directory/'report.html'
