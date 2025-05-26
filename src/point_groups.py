import json
import numpy as np
import os
import sys # 添加 sys 模块

# 从 JSON 文件加载点群数据
def load_point_group_data(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)['point_groups']
    except Exception as e:
        print(f"加载点群数据时出错: {e}")
        # 返回一个空字典作为备用
        return {}

def get_data_file_path(filename):
    """ Get absolute path to data file, works for dev and for PyInstaller """
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Running in a PyInstaller bundle
        # The 'data' directory should be at the root of the bundle (sys._MEIPASS)
        return os.path.join(sys._MEIPASS, 'data', filename)
    else:
        # Running in a normal Python environment
        # Assumes point_groups.py is in 'src' and 'data' is in project_root ('../data')
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # project root
        return os.path.join(base_dir, 'data', filename)

# 尝试加载点群数据
try:
    data_file_actual_path = get_data_file_path('point_group_data.json')
    if not os.path.exists(data_file_actual_path):
        # Fallback for older structure or direct execution within src without data at ../data
        # This path assumes 'data' is a sibling of 'src', and this script is in 'src'
        # ProjectRoot/
        #  |- src/point_groups.py
        #  |- data/point_group_data.json
        # This fallback might be useful if the primary get_data_file_path logic for dev is too strict.
        # However, the get_data_file_path should ideally be robust.
        # For PyInstaller, sys._MEIPASS + 'data/file' is standard if --add-data "data:data" is used.
        # For dev, os.path.join(os.path.dirname(__file__), '..', 'data', filename) from point_groups.py is correct.
        # The get_data_file_path has been updated to use os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # which should give the project root in dev.
        print(f"Warning: Data file not found at primary path: {data_file_actual_path}")
        # Trying alternative relative path for dev (../data from script location)
        alternative_dev_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'point_group_data.json')
        if os.path.exists(alternative_dev_path):
            data_file_actual_path = alternative_dev_path
            print(f"Found data file at alternative dev path: {data_file_actual_path}")
        else:
            print(f"Error: Data file 'point_group_data.json' not found at expected locations.")
            raise FileNotFoundError("point_group_data.json not found")
            
    point_group_components = load_point_group_data(data_file_actual_path)
except Exception as e:
    print(f"初始化点群数据时出错: {e}")
    point_group_components = {}

def str_to_indices(comp_str):
    """将'xyz'类字符串转换为(0,1,2)格式的整数元组"""
    try:
        mapping = {'x':0, 'y':1, 'z':2}
        return tuple(mapping[c] for c in comp_str.lower())
    except Exception as e:
        print(f"转换索引时出错 ({comp_str}): {e}")
        # 返回一个默认值
        return (0, 0, 0)

def create_tensor(components, dim=3):
    """根据字符串分量创建张量"""
    try:
        tensor = np.zeros((dim, dim, dim))
        
        # 处理特殊情况：所有元素都非零
        if len(components) == 1 and components[0] == "All elements are independent and nonzero":
            # 将所有元素设为1.0
            for i in range(dim):
                for j in range(dim):
                    for k in range(dim):
                        tensor[i, j, k] = 1.0
            return tensor
        
        # 处理空列表情况
        if len(components) == 0:
            return tensor
        
        # 正常处理非零分量
        for comp_str in components:
            try:
                indices = str_to_indices(comp_str)
                tensor[indices] = 1.0
            except Exception as e:
                print(f"处理分量 {comp_str} 时出错: {e}")
        
        return tensor
    except Exception as e:
        print(f"创建张量时出错: {e}")
        # 返回一个默认张量
        return np.zeros((dim, dim, dim))

def apply_relations(tensor, relations, dim=3):
    """应用点群的关系约束"""
    try:
        if not relations:
            return tensor
        
        # 解析关系并应用
        for relation in relations:
            try:
                parts = relation.split('=')
                if len(parts) < 2:
                    continue
                    
                # 获取第一个分量的索引和值
                first_comp = parts[0].strip()
                first_indices = str_to_indices(first_comp)
                first_value = tensor[first_indices]
                
                # 应用到其他分量
                for i in range(1, len(parts)):
                    comp = parts[i].strip()
                    # 处理负号
                    sign = 1.0
                    if comp.startswith('-'):
                        sign = -1.0
                        comp = comp[1:].strip()
                        
                    indices = str_to_indices(comp)
                    tensor[indices] = sign * first_value
            except Exception as e:
                print(f"应用关系 {relation} 时出错: {e}")
        
        return tensor
    except Exception as e:
        print(f"应用关系约束时出错: {e}")
        return tensor

def get_all_point_groups():
    """返回所有可用的点群列表"""
    try:
        return list(point_group_components.keys())
    except Exception as e:
        print(f"获取点群列表时出错: {e}")
        return []

def get_components_for_group(group_name):
    """获取指定点群的非零分量列表"""
    try:
        if group_name not in point_group_components:
            print(f"警告: 点群 {group_name} 不存在")
            return []
            
        components = point_group_components[group_name]['Non-zero components']
        
        # 处理特殊情况
        if len(components) == 1 and components[0] == "All elements are independent and nonzero":
            # 生成所有可能的分量
            all_components = []
            for i in ['x', 'y', 'z']:
                for j in ['x', 'y', 'z']:
                    for k in ['x', 'y', 'z']:
                        all_components.append(i+j+k)
            return all_components
        
        return components
    except Exception as e:
        print(f"获取点群 {group_name} 的分量时出错: {e}")
        return []

def get_relations_for_group(group_name):
    """获取指定点群的关系约束"""
    try:
        if group_name not in point_group_components:
            return []
            
        if 'Relations' in point_group_components[group_name]:
            return point_group_components[group_name]['Relations']
        return []
    except Exception as e:
        print(f"获取点群 {group_name} 的关系约束时出错: {e}")
        return []

def create_tensor_with_relations(group_name, dim=3):
    """创建包含关系约束的张量"""
    try:
        components = get_components_for_group(group_name)
        relations = get_relations_for_group(group_name)
        
        # 先创建基本张量
        tensor = create_tensor(components, dim)
        
        # 应用关系约束
        if relations:
            tensor = apply_relations(tensor, relations, dim)
        
        return tensor
    except Exception as e:
        print(f"创建带关系的张量时出错: {e}")
        # 返回一个默认张量
        return np.zeros((dim, dim, dim))

def show_component_table(point_group):
    """显示指定点群的非零分量"""
    try:
        # 检查点群是否存在
        if point_group not in point_group_components:
            print(f"警告: 点群 {point_group} 不存在于点群组件数据中")
            return
            
        # 获取非零分量列表
        if 'Non-zero components' not in point_group_components[point_group]:
            print(f"警告: 点群 {point_group} 缺少非零分量数据")
            return
            
        components = point_group_components[point_group]['Non-zero components']
        
        # 处理特殊情况：空列表
        if len(components) == 0:
            print(f"\n{point_group}的非线性极化张量分量全部为零")
            return
        
        # 处理特殊情况：所有元素非零
        if len(components) == 1 and components[0] == "All elements are independent and nonzero":
            print(f"\n{point_group}的非线性极化张量所有分量都是独立的非零值")
            return
            
        # 正常情况：打印所有非零分量
        print(f"\n{point_group}的非零非线性极化张量分量：")
        for i, comp in enumerate(components, 1):
            print(f"{i}. χ{comp}")
        
        # 显示关系约束（如果存在）
        try:
            relations = get_relations_for_group(point_group)
            if relations:
                print("\n分量之间的关系：")
                for i, relation in enumerate(relations, 1):
                    print(f"{i}. χ{relation}")
        except Exception as e:
            print(f"获取关系约束时出错: {e}")
            
    except Exception as e:
        print(f"显示点群 {point_group} 的分量表时出错: {e}")
        print(f"异常详情: {str(e)}")