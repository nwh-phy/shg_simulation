"""Reproducible command-line entry for the same experimental import functions as the GUI."""
import argparse
import sys

from experimental_data import (
    ExtractionSettings, load_angle_curve, load_spectral_table,
    load_spectrum_folder, extract_curve, export_curve,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="导入或提取 SHG 数据；角度为已换算的入射偏振角。")
    parser.add_argument("input", help="输入文件或单个测量序列目录")
    parser.add_argument("--kind", choices=("curve", "table", "folder"), required=True)
    parser.add_argument("--output", required=True, help="新建导出目录，已存在时拒绝覆盖")
    parser.add_argument("--target-nm", type=float)
    parser.add_argument("--step-deg", type=float, help="目录模式必填，没有默认步长")
    parser.add_argument("--start-deg", type=float, default=0)
    parser.add_argument("--method", choices=("nearest", "linear", "window_max", "integral"), default="nearest")
    parser.add_argument("--half-width-nm", type=float, default=1)
    parser.add_argument("--max-distance-nm", type=float, default=0.5)
    parser.add_argument("--background", type=float, nargs=2, action="append", default=[], metavar=("LOW", "HIGH"))
    parser.add_argument("--channel", choices=("unspecified", "parallel", "perpendicular"), default="unspecified")
    parser.add_argument('--background-model', choices=('constant', 'linear'), default='constant')
    parser.add_argument('--exposure-s', type=float)
    parser.add_argument('--power-mw', type=float)
    parser.add_argument('--saturation-level', type=float)
    args = parser.parse_args(argv)
    if args.kind != "curve" and args.target_nm is None:
        parser.error("原始光谱模式必须指定 --target-nm。")
    if args.kind == "folder" and args.step_deg is None:
        parser.error("目录模式必须手动指定 --step-deg，没有默认步长。")
    try:
        if args.kind == "curve":
            curve = load_angle_curve(args.input)
        else:
            settings = ExtractionSettings(args.target_nm, args.method, args.half_width_nm, args.max_distance_nm, tuple(tuple(r) for r in args.background), args.background_model, args.exposure_s, args.power_mw, args.saturation_level)
            if args.kind == "folder":
                spectra, metadata = load_spectrum_folder(args.input, args.step_deg, args.start_deg)
            else:
                spectra, metadata = load_spectral_table(args.input)
            curve = extract_curve(spectra, settings, metadata)
        if args.channel != "unspecified":
            curve.metadata["channel"] = args.channel
        paths = export_curve(curve, args.output)
    except (OSError, ValueError, TypeError) as error:
        print(f"导入失败：{error}", file=sys.stderr)
        return 1
    for message in curve.metadata.get("warnings", []):
        print(message)
    print(f"成功导出 {len(curve.angle_deg)} 条记录：{paths[0]}")
    print(f"处理记录：{paths[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
