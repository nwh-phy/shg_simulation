"""Reproducible joint-fit example; no experimental data or inferred material identity."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from joint_fit import make_demo, fit_joint, FitOptions

if __name__ == '__main__':
    curves, group, geometry = make_demo()
    result = fit_joint(curves, group, geometry, options=FitOptions(bootstrap=100))
    print('SYNTHETIC EXAMPLE, true orientation = 17.3 deg')
    print(result['parameters'])
    print(result['intervals_95'])
    print(result['warnings'])
