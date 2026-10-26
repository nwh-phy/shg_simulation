"""
常量定义模块
"""

# Voigt索引映射: (j, k) -> Voigt列索引
VOIGT_INDEX_MAP = {
    (0, 0): 0,  # xx -> 1
    (1, 1): 1,  # yy -> 2
    (2, 2): 2,  # zz -> 3
    (1, 2): 3, (2, 1): 3,  # yz -> 4
    (0, 2): 4, (2, 0): 4,  # xz -> 5
    (0, 1): 5, (1, 0): 5,  # xy -> 6
}

# 坐标字符到索引的映射
COORD_TO_INDEX = {'x': 0, 'y': 1, 'z': 2}
INDEX_TO_COORD = {0: 'x', 1: 'y', 2: 'z'}

# Voigt列索引到d_ij标签的映射
VOIGT_TO_LABEL = {
    0: '1',  # xx
    1: '2',  # yy
    2: '3',  # zz
    3: '4',  # yz
    4: '5',  # xz
    5: '6',  # xy
}


def str_to_indices(comp_str: str) -> tuple:
    """将'xyz'类字符串转换为(0,1,2)格式的整数元组"""
    return tuple(COORD_TO_INDEX[c] for c in comp_str.lower())


def indices_to_str(indices: tuple) -> str:
    """将(0,1,2)格式的整数元组转换为'xyz'类字符串"""
    return ''.join(INDEX_TO_COORD[i] for i in indices)


def get_voigt_label(component_str: str) -> str:
    """将xyz格式转换为d_ij Voigt标签格式"""
    indices = str_to_indices(component_str)
    i = indices[0]
    jk = (indices[1], indices[2])
    l = VOIGT_INDEX_MAP.get(jk, -1)
    if l >= 0:
        return f"d{i+1}{VOIGT_TO_LABEL[l]}"
    return f"d{indices[0]+1}{indices[1]+1}{indices[2]+1}"
