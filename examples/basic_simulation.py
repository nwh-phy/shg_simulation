"""Canonical normal-incidence D3h demo; generated data only."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import matplotlib.pyplot as plt
from point_groups import get_all_point_groups
from reflection_model import predict

if __name__ == '__main__':
    group = next(g for g in get_all_point_groups() if g.startswith('6̄m2 ='))
    angles = np.linspace(0, 360, 721)
    for channel in ['parallel', 'perpendicular']:
        plt.plot(angles, predict(group, angles, 17.3, {'yyy': 1}, channel=channel), label=channel)
    plt.xlabel('Incident polarization angle (deg)'); plt.ylabel('Relative intensity (synthetic)')
    plt.legend(); plt.show()
