"""
模拟状态数据模型
"""
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np


@dataclass
class CrystalOrientation:
    """晶体取向参数（欧拉角）"""
    phi_c: float = 0.0    # Z轴旋转（度）
    theta_c: float = 0.0  # Y'轴旋转（度）
    psi_c: float = 0.0    # Z''轴旋转（度）

    def is_identity(self) -> bool:
        """检查是否为单位旋转"""
        return self.phi_c == 0 and self.theta_c == 0 and self.psi_c == 0


@dataclass
class SimulationState:
    """模拟状态 - 单一数据源"""
    # 点群信息
    point_group_name: str = ""
    base_tensor: Optional[np.ndarray] = None

    # 分量系数
    component_values: Dict[str, float] = field(default_factory=dict)
    global_scale: float = 1.0

    # 晶体取向
    orientation: CrystalOrientation = field(default_factory=CrystalOrientation)

    # 扫描参数
    scan_mode: str = "alpha_polar"
    phi_beam: float = 0.0
    theta_fixed: float = 0.0
    alpha_polarization: float = 0.0

    # 偏振设置
    input_polarization_mode: str = "linear"
    detection_mode: int = 0  # 0=Total, 1=Parallel, 2=Perpendicular
