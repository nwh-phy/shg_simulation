"""
张量操作模块 - 向量化实现
"""
import numpy as np
from typing import Tuple


class TensorOperations:
    """张量操作工具类"""

    @staticmethod
    def get_rotation_matrix(phi_c: float, theta_c: float, psi_c: float) -> np.ndarray:
        """
        计算ZYZ欧拉角旋转矩阵

        Args:
            phi_c: Z轴旋转角度（度）
            theta_c: Y'轴旋转角度（度）
            psi_c: Z''轴旋转角度（度）

        Returns:
            3x3旋转矩阵
        """
        phi_rad = np.deg2rad(phi_c)
        theta_rad = np.deg2rad(theta_c)
        psi_rad = np.deg2rad(psi_c)

        # Z轴旋转 (phi_c)
        Rz_phi = np.array([
            [np.cos(phi_rad), -np.sin(phi_rad), 0],
            [np.sin(phi_rad), np.cos(phi_rad), 0],
            [0, 0, 1]
        ])

        # Y'轴旋转 (theta_c)
        Ry_theta = np.array([
            [np.cos(theta_rad), 0, np.sin(theta_rad)],
            [0, 1, 0],
            [-np.sin(theta_rad), 0, np.cos(theta_rad)]
        ])

        # Z''轴旋转 (psi_c)
        Rz_psi = np.array([
            [np.cos(psi_rad), -np.sin(psi_rad), 0],
            [np.sin(psi_rad), np.cos(psi_rad), 0],
            [0, 0, 1]
        ])

        return Rz_phi @ Ry_theta @ Rz_psi

    @staticmethod
    def rotate_tensor_vectorized(tensor_crystal: np.ndarray, R: np.ndarray) -> np.ndarray:
        """
        向量化张量旋转 - 使用np.einsum替代6层循环

        将3阶张量从晶体坐标系变换到实验室坐标系:
        d'_pqr = R_pi * R_qj * R_rk * d_ijk

        Args:
            tensor_crystal: 3x3x3晶体坐标系张量
            R: 3x3旋转矩阵

        Returns:
            3x3x3实验室坐标系张量
        """
        return np.einsum('pi,qj,rk,ijk->pqr', R, R, R, tensor_crystal)

    @staticmethod
    def tensor_to_voigt(tensor_3x3x3: np.ndarray) -> np.ndarray:
        """
        将3x3x3张量转换为3x6 Voigt矩阵

        Voigt映射:
        - 列0: xx (j=0,k=0)
        - 列1: yy (j=1,k=1)
        - 列2: zz (j=2,k=2)
        - 列3: yz (j=1,k=2)
        - 列4: xz (j=0,k=2)
        - 列5: xy (j=0,k=1)

        Args:
            tensor_3x3x3: 3x3x3张量

        Returns:
            3x6 Voigt矩阵
        """
        d_matrix = np.zeros((3, 6), dtype=np.result_type(tensor_3x3x3.dtype, float))
        d_matrix[:, 0] = tensor_3x3x3[:, 0, 0]  # xx
        d_matrix[:, 1] = tensor_3x3x3[:, 1, 1]  # yy
        d_matrix[:, 2] = tensor_3x3x3[:, 2, 2]  # zz
        d_matrix[:, 3] = (tensor_3x3x3[:, 1, 2] + tensor_3x3x3[:, 2, 1]) / 2
        d_matrix[:, 4] = (tensor_3x3x3[:, 0, 2] + tensor_3x3x3[:, 2, 0]) / 2
        d_matrix[:, 5] = (tensor_3x3x3[:, 0, 1] + tensor_3x3x3[:, 1, 0]) / 2
        return d_matrix
