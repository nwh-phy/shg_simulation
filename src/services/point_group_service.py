"""
点群数据服务
"""
import numpy as np
from typing import List

# 导入现有的point_groups模块（保持API兼容）
from point_groups import (
    get_all_point_groups,
    get_components_for_group,
    create_tensor_with_relations,
    str_to_indices
)


class PointGroupService:
    """点群数据服务 - 封装point_groups.py"""

    def get_all_groups(self) -> List[str]:
        """获取所有可用点群"""
        return get_all_point_groups()

    def get_components(self, group_name: str) -> List[str]:
        """获取点群的非零分量"""
        return get_components_for_group(group_name)

    def create_tensor(self, group_name: str) -> np.ndarray:
        """创建点群的张量"""
        return create_tensor_with_relations(group_name)

    @staticmethod
    def str_to_indices(comp_str: str) -> tuple:
        """将分量字符串转换为索引"""
        return str_to_indices(comp_str)
