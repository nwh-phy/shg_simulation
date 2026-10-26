"""
测试新架构模块
"""
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_core_modules():
    """测试核心计算模块"""
    print("测试核心计算模块...")

    import numpy as np
    from core.tensor_operations import TensorOperations
    from core.shg_calculator import (
        SHGCalculator, ScanParameters,
        PolarizationMode, DetectionMode
    )

    # 测试张量旋转
    tensor = np.zeros((3, 3, 3))
    tensor[2, 2, 2] = 1.0  # zzz分量

    R = TensorOperations.get_rotation_matrix(0, 0, 0)
    rotated = TensorOperations.rotate_tensor_vectorized(tensor, R)

    assert rotated.shape == (3, 3, 3), "张量形状错误"
    print("  [OK] 张量旋转测试通过")

    # 测试Voigt转换
    d_matrix = TensorOperations.tensor_to_voigt(tensor)
    assert d_matrix.shape == (3, 6), "Voigt矩阵形状错误"
    print("  [OK] Voigt转换测试通过")

    # 测试SHG计算
    params = ScanParameters(
        theta_range=np.linspace(0, 2*np.pi, 36),
        phi_fixed=0.0,
        alpha_fixed=0.0
    )
    calculator = SHGCalculator(d_matrix)
    intensities = calculator.calculate_intensity_vectorized(params)

    assert len(intensities) == 36, "强度数组长度错误"
    print("  [OK] SHG计算测试通过")

    print("核心模块测试完成!\n")

def test_models():
    """测试数据模型"""
    print("测试数据模型...")

    from models.simulation_state import SimulationState, CrystalOrientation
    from models.crystal_presets import COMMON_CRYSTALS

    # 测试状态创建
    state = SimulationState()
    assert state.point_group_name == "", "默认点群名称错误"
    print("  [OK] SimulationState创建测试通过")

    # 测试晶体取向
    orientation = CrystalOrientation(phi_c=45.0)
    assert not orientation.is_identity(), "取向判断错误"
    print("  [OK] CrystalOrientation测试通过")

    # 测试预设
    assert "LiNbO3" in COMMON_CRYSTALS, "预设晶体缺失"
    print("  [OK] 晶体预设测试通过")

    print("数据模型测试完成!\n")

if __name__ == "__main__":
    print("=" * 50)
    print("SHG Simulator 新架构测试")
    print("=" * 50 + "\n")

    try:
        test_core_modules()
        test_models()
        print("=" * 50)
        print("所有测试通过!")
        print("=" * 50)
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
