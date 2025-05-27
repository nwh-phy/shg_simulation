import sympy
import os
import sys

# 定义符号
alpha = sympy.symbols('alpha') # 入射光偏振角
E0 = sympy.symbols('E0')       # 入射光电场振幅

# 定义入射光电场分量 (假设垂直入射，沿Z轴传播，偏振角alpha为与X轴夹角)
Ex = E0 * sympy.cos(alpha)
Ey = E0 * sympy.sin(alpha)
Ez = sympy.sympify(0) # 显式定义为sympy的0

E_vector_lab = sympy.Matrix([Ex, Ey, Ez])

# 从 point_groups.py 导入必要的函数
# 注意：这可能需要调整PYTHONPATH或者采用更健壮的导入方式，取决于项目结构和运行方式
# 为了简单起见，我们先假设可以直接导入
try:
    from point_groups import load_point_group_data, get_data_file_path, str_to_indices as pg_str_to_indices
except ImportError:
    # Fallback for cases where direct import might fail (e.g. running script standalone)
    # This might need adjustment based on your project structure.
    sys.path.append(os.path.dirname(os.path.abspath(__file__))) # Add current dir
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')) # Add parent dir (project root if src is a subdir)
    from point_groups import load_point_group_data, get_data_file_path, str_to_indices as pg_str_to_indices

# 加载点群数据 (全局加载一次)
# TODO: Handle potential errors more gracefully if point_group_data.json is missing
POINT_GROUP_DATA = {}
try:
    data_file = get_data_file_path('point_group_data.json')
    POINT_GROUP_DATA = load_point_group_data(data_file)
except Exception as e:
    print(f"Error loading point group data in symbolic_calculator: {e}")
    # Handle the absence of data, perhaps by disabling features or raising an error

def get_symbolic_d_tensor(point_group_name):
    """
    Creates a symbolic 2nd order NLO tensor (d_ijk) for a given crystal point group.
    The tensor elements will be sympy.Symbols, respecting the point group's symmetry.
    Uses a 3-pass approach for compatibility with current JSON structure.

    Args:
        point_group_name (str): The name of the point group (e.g., "3m", "4mm").

    Returns:
        sympy.Array: A 3x3x3 symbolic tensor.
                     Returns a tensor of zeros if point group not found or has no components.
    """
    if not POINT_GROUP_DATA or point_group_name not in POINT_GROUP_DATA:
        print(f"Point group '{point_group_name}' not found in loaded data.")
        return sympy.MutableDenseNDimArray.zeros(3,3,3)

    group_info = POINT_GROUP_DATA[point_group_name]
    non_zero_components_str_list = group_info.get('Non-zero components', [])
    relations_str_list = group_info.get('Relations', [])

    d_tensor_sym = sympy.MutableDenseNDimArray.zeros(3,3,3)
    # Map from component string (e.g. "zzz") to its sympy.Symbol
    # This map will be updated by relations.
    component_to_symbol_map = {}

    # Handle special case: "All elements are independent and nonzero"
    if len(non_zero_components_str_list) == 1 and non_zero_components_str_list[0] == "All elements are independent and nonzero":
        coord_map = ['x', 'y', 'z']
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    comp_str = f"{coord_map[i]}{coord_map[j]}{coord_map[k]}"
                    sym_name = f"d_{comp_str}"
                    sym = sympy.symbols(sym_name)
                    # component_to_symbol_map[comp_str] = sym # Not strictly needed here as no relations
                    d_tensor_sym[i, j, k] = sym
        if relations_str_list:
            print(f"Warning: Point group '{point_group_name}' is specified as all non-zero and independent, but also has relations defined. Relations will be ignored.")
        return sympy.Array(d_tensor_sym)

    # Pass 1: Create initial symbols for all components listed in "Non-zero components"
    # These are treated as potentially independent until relations are processed.
    for comp_str in non_zero_components_str_list:
        if not comp_str: continue
        if comp_str not in component_to_symbol_map: # Should always be true here unless duplicate in list
            component_to_symbol_map[comp_str] = sympy.symbols(f"d_{comp_str}")

    # Pass 2: Process relations to alias symbols in component_to_symbol_map
    # The first component in a relation string (e.g., 'LHS' in 'LHS = RHS1 = -RHS2')
    # is considered the "master" symbol for that relation.
    for rel_str in relations_str_list:
        if not rel_str: continue
        
        parts = rel_str.split('=')
        if len(parts) < 2:
            print(f"Warning: Malformed relation string '{rel_str}' for point group '{point_group_name}'. Skipping.")
            continue

        # First component in the full relation string is the base/master for this relation
        first_comp_overall_str_with_sign = parts[0].strip()
        base_sign = 1
        base_comp_str = first_comp_overall_str_with_sign.lstrip('-').strip()
        if first_comp_overall_str_with_sign.startswith('-'):
            # This is unusual for the very first component of a relation to define the base.
            # e.g. "-d1 = d2". It implies d1 is base, and d2 = -d1.
            # Or, if d1 is base, its symbol is d_d1, and the relation means values are negative.
            # For now, assume the symbol itself is "d_name", and sign applies to its value.
            # Let's assume the symbol name itself doesn't carry the sign.
            # The first term defines the symbol all others map to.
            # print(f"Warning: First component '{first_comp_overall_str_with_sign}' in relation '{rel_str}' starts with a sign. The symbol name will not include the sign.")
            # base_sign = -1 # This interpretation is tricky.
            # base_comp_str = first_comp_overall_str_with_sign[1:].strip()
            pass # Sign of the base component is handled when it's used as a source.

        # Ensure the base component string is valid and has a symbol
        if not base_comp_str:
            print(f"Warning: Empty base component string in relation '{rel_str}' for {point_group_name}. Skipping.")
            continue
        
        # The symbol for the base component. It should exist from Pass 1 if JSON is consistent.
        # If not in "Non-zero components" but starts a relation, it becomes an independent symbol.
        if base_comp_str not in component_to_symbol_map:
            print(f"Warning: Base component '{base_comp_str}' of relation '{rel_str}' for {point_group_name} was not in 'Non-zero components'. Creating symbol for it.")
            component_to_symbol_map[base_comp_str] = sympy.symbols(f"d_{base_comp_str}")
        
        current_base_symbol_obj = component_to_symbol_map[base_comp_str]
        # Apply the initial sign of the base term if it was negative (e.g. relation like "-d1 = d2")
        # This means d2 will be set to (-1 * symbol_of_d1)
        # This logic is subtle: if -d1 = d2, then d1's symbol is S1, d2's symbol becomes -S1
        if first_comp_overall_str_with_sign.startswith('-'):
             master_value_for_relation = -current_base_symbol_obj
        else:
             master_value_for_relation = current_base_symbol_obj


        # All other components in the relation string get mapped to this master_value_for_relation
        for i in range(len(parts)): # Iterate through all parts of "LHS = RHS1 = RHS2"
            # The first part (parts[0]) established the master_value_for_relation.
            # Subsequent parts (parts[1], parts[2], ...) are equated to it.
            # But a relation is "A = B = C". parts[0]=A, parts[1]=B, parts[2]=C.
            # A is master. B maps to A. C maps to A.
            # If relation is "A = B", parts[0]=A, parts[1]=B. B maps to A.
            
            comp_def_str_with_sign = parts[i].strip()
            if i == 0 : # This is the master component itself, its symbol is already set in the map.
                # We just need to ensure its map entry is the positive symbol if it was like "-d1=d2"
                # component_to_symbol_map[base_comp_str] is already current_base_symbol_obj
                continue 

            # These are the dependent components
            sign = 1
            comp_str_rel = comp_def_str_with_sign
            if comp_def_str_with_sign.startswith('-'):
                sign = -1
                comp_str_rel = comp_def_str_with_sign[1:].strip()

            if not comp_str_rel:
                print(f"Warning: Empty dependent component string in relation '{rel_str}' for {point_group_name}. Skipping term.")
                continue

            # If this component was not in "Non-zero components", it might not be in map.
            # This implies a component defined *only* by a relation.
            # It should still map to the master symbol of the relation.
            if comp_str_rel not in component_to_symbol_map:
                 print(f"Note: Component '{comp_str_rel}' in relation '{rel_str}' for {point_group_name} was not in 'Non-zero components'. It will be defined by this relation.")
            
            # The value this component (comp_str_rel) takes is based on master_value_for_relation.
            # If relation is A = B, master is A (value S_A). B gets S_A. comp_str_rel is B. sign is 1.
            # If relation is A = -B, master is A (value S_A). B gets -S_A. comp_str_rel is B. sign is -1.
            # So, component_to_symbol_map[comp_str_rel] = (1/sign) * master_value_for_relation
            # This is equivalent to: if A=B, B=A. if A=-B, -B=A => B=-A.
            if sign == 1:
                component_to_symbol_map[comp_str_rel] = master_value_for_relation
            else: # sign == -1
                component_to_symbol_map[comp_str_rel] = -master_value_for_relation


    # Pass 3: Populate the tensor using the (potentially aliased) symbols from component_to_symbol_map
    # Only components originally listed in "Non-zero components" will be placed in the tensor.
    # If a component was ONLY defined in a relation (not in non-zero list), it won't be added to tensor here.
    # This is typically correct as "Non-zero components" should be exhaustive for what *can* be non-zero.
    for comp_str in non_zero_components_str_list:
        if not comp_str: continue
        
        final_symbol_for_comp = component_to_symbol_map.get(comp_str)
        if final_symbol_for_comp is None:
            # This should not happen if comp_str was in non_zero_components_str_list and Pass 1 worked.
            print(f"Error: Symbol for '{comp_str}' (from non-zero list) not found in map for {point_group_name}. Setting to 0.")
            final_symbol_for_comp = sympy.S.Zero
            
        indices = pg_str_to_indices(comp_str)
        # Check if already set by a more primary definition (should not happen with 3-pass if map is source)
        # if d_tensor_sym[indices] != 0 and d_tensor_sym[indices] != final_symbol_for_comp:
        #    print(f"Internal Warning: Overwriting tensor at {indices} for {comp_str} in {point_group_name}. Was {d_tensor_sym[indices]}, new {final_symbol_for_comp}")
        d_tensor_sym[indices] = final_symbol_for_comp
        
    return sympy.Array(d_tensor_sym)

def calculate_symbolic_shg_expressions(point_group_name, crystal_orientation_info=None):
    """
    Calculates symbolic expressions for SHG intensity (total, parallel, perpendicular)
    and the polarization vector components (Px, Py, Pz) for a given point group.

    Args:
        point_group_name (str): The name of the point group.
        crystal_orientation_info (dict, optional): Information about crystal orientation.
                                                 Currently ignored; assumes tensor is in lab frame.

    Returns:
        dict: A dictionary containing symbolic expressions for Px, Py, Pz,
              I_total, I_parallel, I_perpendicular.
              Returns None if the point group results in an all-zero d_tensor.
    """
    d_sym = get_symbolic_d_tensor(point_group_name)

    # Create a string representation of the 3x3x3 d_sym tensor for inspection if needed
    # This can be removed or changed if d_voigt_matrix_str is preferred as the primary tensor string output
    # original_d_tensor_str = sympy.pretty(d_sym, use_unicode=False, wrap_line=False)

    # --- Convert 3x3x3 d_sym to 3x6 Voigt Matrix d_voigt_matrix ---
    d_voigt_matrix = sympy.zeros(3, 6)
    # Mapping: d_il where i is 1,2,3 and l is 1..6
    # l=1 -> jk=11 (00 in 0-indexed)
    # l=2 -> jk=22 (11)
    # l=3 -> jk=33 (22)
    # l=4 -> jk=23 or 32 (12 or 21) - Standard is d_i4 = d_i23 (symmetric in last two for d_eff)
    # l=5 -> jk=13 or 31 (02 or 20) - Standard is d_i5 = d_i13
    # l=6 -> jk=12 or 21 (01 or 10) - Standard is d_i6 = d_i12
    
    # d_i1 = d_i11 -> d_sym[i,0,0]
    # d_i2 = d_i22 -> d_sym[i,1,1]
    # d_i3 = d_i33 -> d_sym[i,2,2]
    # d_i4 = d_i23 (or d_i32) -> d_sym[i,1,2] (assuming Kleinman, or just one is non-zero by symmetry)
    # d_i5 = d_i13 (or d_i31) -> d_sym[i,0,2]
    # d_i6 = d_i12 (or d_i21) -> d_sym[i,0,1]

    for i in range(3):
        d_voigt_matrix[i, 0] = d_sym[i, 0, 0]  # xx
        d_voigt_matrix[i, 1] = d_sym[i, 1, 1]  # yy
        d_voigt_matrix[i, 2] = d_sym[i, 2, 2]  # zz
        # For off-diagonal elements in Voigt notation, d_il, we sum d_ijk and d_ikj if not already symmetric.
        # However, d_sym from get_symbolic_d_tensor already respects symmetries where d_ijk = d_ikj (Kleinman)
        # or where only one is non-zero. The point group data primarily defines unique non-zero d_ijk.
        # If Kleinman (d_ijk = d_ikj) is assumed or implied by point group data, then:
        # d_i4 maps to d_i23 (or d_i32). If they are distinct symbols, this needs care.
        # For now, let's assume d_ijk where j and k are the Voigt indices.
        # The mapping used in main.py's _calculate_lab_frame_d_matrix is direct:
        # d_matrix[:, 3] = tensor_lab_frame[:, 1, 2] (maps to d_i23)
        # d_matrix[:, 4] = tensor_lab_frame[:, 0, 2] (maps to d_i13)
        # d_matrix[:, 5] = tensor_lab_frame[:, 0, 1] (maps to d_i12)
        # We will follow this direct mapping for consistency with numerical part.
        d_voigt_matrix[i, 3] = d_sym[i, 1, 2]  # yz (or 23)
        d_voigt_matrix[i, 4] = d_sym[i, 0, 2]  # xz (or 13)
        d_voigt_matrix[i, 5] = d_sym[i, 0, 1]  # xy (or 12)

    d_voigt_matrix_str = sympy.pretty(d_voigt_matrix, use_unicode=False, wrap_line=False)

    # Check if d_voigt_matrix is effectively zero (this check can also be done on d_sym)
    is_zero_tensor = all(d_voigt_matrix[i,j] == 0 for i in range(3) for j in range(6))
    if is_zero_tensor:
        zero_expr = sympy.S.Zero
        return {
            'd_tensor_str': sympy.pretty(sympy.zeros(3,6), use_unicode=False, wrap_line=False),
            'd_voigt_matrix_obj': sympy.zeros(3,6),
            'Px': zero_expr, 'Py': zero_expr, 'Pz': zero_expr,
            'I_total': zero_expr, 'I_parallel': zero_expr, 'I_perpendicular': zero_expr
        }

    # Symbolic electric field in lab frame (already defined globally)
    E_vec = E_vector_lab

    # Calculate symbolic polarization P_i = sum_jk d_ijk E_j E_k
    Px_sym = sympy.S.Zero
    Py_sym = sympy.S.Zero
    Pz_sym = sympy.S.Zero

    for j_idx in range(3):
        for k_idx in range(3):
            Ej_Ek_term = E_vec[j_idx] * E_vec[k_idx]
            if Ej_Ek_term == sympy.S.Zero: # Optimization: if Ej*Ek is zero, skip
                continue
            
            if d_sym[0, j_idx, k_idx] != 0:
                Px_sym += d_sym[0, j_idx, k_idx] * Ej_Ek_term
            if d_sym[1, j_idx, k_idx] != 0:
                Py_sym += d_sym[1, j_idx, k_idx] * Ej_Ek_term
            if d_sym[2, j_idx, k_idx] != 0:
                Pz_sym += d_sym[2, j_idx, k_idx] * Ej_Ek_term
    
    Px_sym = sympy.trigsimp(Px_sym)
    Py_sym = sympy.trigsimp(Py_sym)
    Pz_sym = sympy.trigsimp(Pz_sym)

    P_vector_sym = sympy.Matrix([Px_sym, Py_sym, Pz_sym])

    # Calculate SHG Intensities
    # I_total = |P|^2 = Px^2 + Py^2 + Pz^2 (assuming Px,Py,Pz are real amplitudes here for simplicity, 
    # or |Px|^2 + |Py|^2 + |Pz|^2 if they can be complex)
    # For symbolic expressions, we usually don't take conjugates unless E0 or d_ijk are complex.
    # Let's assume E0 and d_ijk symbols are real for now.
    I_total_sym = Px_sym**2 + Py_sym**2 + Pz_sym**2
    I_total_sym = sympy.trigsimp(I_total_sym)

    # I_parallel: Intensity component parallel to incident polarization E_in = [Ex, Ey, 0]
    # P_parallel_component = (P . E_in_normalized_xy_plane)
    # E_in_xy_normalized_approx for symbolic (assuming E0 is not zero):
    # Ex_norm_symbolic = sympy.cos(alpha)
    # Ey_norm_symbolic = sympy.sin(alpha)
    # P_dot_Ein_norm = Px_sym * Ex_norm_symbolic + Py_sym * Ey_norm_symbolic
    # I_parallel_sym = P_dot_Ein_norm**2 
    
    # More robustly, E_in_xy_plane is [E0*cos(a), E0*sin(a), 0]
    # Let e_parallel_vec_xy = [cos(a), sin(a)] be the direction of incident E-field in xy plane.
    # P_parallel_val = Px_sym * sympy.cos(alpha) + Py_sym * sympy.sin(alpha)
    # I_parallel_sym = P_parallel_val**2
    # This is correct assuming P_vector_sym is the SHG polarization in the lab frame,
    # and we are analyzing its component along the fundamental's E-field direction in the XY plane.

    # Direction of incident polarization in XY plane (normalized, assuming E0!=0)
    e_parallel_x = sympy.cos(alpha)
    e_parallel_y = sympy.sin(alpha)

    P_parallel_component = Px_sym * e_parallel_x + Py_sym * e_parallel_y
    I_parallel_sym = sympy.trigsimp(P_parallel_component**2)

    # I_perpendicular: Intensity component perpendicular to incident polarization in XY plane.
    # e_perpendicular_vec_xy = [-sin(a), cos(a)]
    # P_perpendicular_component = Px_sym * (-sympy.sin(alpha)) + Py_sym * sympy.cos(alpha)
    # I_perpendicular_sym = P_perpendicular_component**2

    e_perpendicular_x = -sympy.sin(alpha)
    e_perpendicular_y = sympy.cos(alpha)
    
    P_perpendicular_component = Px_sym * e_perpendicular_x + Py_sym * e_perpendicular_y
    I_perpendicular_sym = sympy.trigsimp(P_perpendicular_component**2)
    
    return {
        'd_tensor_str': d_voigt_matrix_str, # String of 3x6 Voigt matrix
        'd_voigt_matrix_obj': d_voigt_matrix, # Actual 3x6 SymPy Matrix object
        # 'original_3x3x3_d_tensor_str': original_d_tensor_str, # Optionally include this too
        'Px': Px_sym,
        'Py': Py_sym,
        'Pz': Pz_sym,
        'I_total': I_total_sym,
        'I_parallel': I_parallel_sym,
        'I_perpendicular': I_perpendicular_sym
    }

# Example usage (for testing within this file if run directly)
if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    print("开始加载点群数据...")
    if not POINT_GROUP_DATA:
        try:
            data_file_main = get_data_file_path('point_group_data.json')
            POINT_GROUP_DATA = load_point_group_data(data_file_main)
            if not POINT_GROUP_DATA:
                raise Exception("POINT_GROUP_DATA 数据为空.")
            else:
                print("点群数据加载成功.")
                # print("可用的点群关键字:") # 取消此行注释以查看所有键
                # for key in POINT_GROUP_DATA.keys():
                #     print(f"  '{key}'")
        except Exception as e_main:
            print(f"加载点群数据失败: {e_main}")
            sys.exit(1)
    else:
        print("点群数据已预加载.")

    if POINT_GROUP_DATA:
        print("\n--- 开始测试点群 '3m = C₃ᵥ (trigonal)' 的符号张量 --- ")
        pg_3m = "3m = C₃ᵥ (trigonal)"
        d_3m = get_symbolic_d_tensor(pg_3m)

        s_zzz, s_zxx, s_xxz, s_xzx, s_yyy = sympy.symbols('d_zzz d_zxx d_xxz d_xzx d_yyy')

        if d_3m is not None and d_3m.shape == (3,3,3) and d_3m != sympy.MutableDenseNDimArray.zeros(3,3,3):
            print(f"为 {pg_3m} 创建张量成功.")
            expected_prints_3m = {
                (2,2,2): s_zzz, (2,0,0): s_zxx, (2,1,1): s_zxx,
                (0,0,2): s_xxz, (1,1,2): s_xxz,
                (1,0,0): -s_yyy,(1,1,1): s_yyy, (0,0,1): -s_yyy, (0,1,0): -s_yyy,
                (0,2,0): s_xzx, (1,2,1): s_xzx
            }
            for idx_tuple, val_expr in expected_prints_3m.items():
                 print(f"d{idx_tuple}: {d_3m[idx_tuple]} (预期: {val_expr})")
            print(f"d(0,0,0): {d_3m[0,0,0]} (预期: 0)")

            valid_3m = True
            for indices, expected_sym in expected_prints_3m.items():
                if d_3m[indices] != expected_sym:
                    print(f"错误: 3m 张量元素 d{indices} 为 {d_3m[indices]}, 预期为: {expected_sym}")
                    valid_3m = False
            
            zero_checks_3m = [(0,0,0), (0,1,2), (1,0,2)]
            for indices in zero_checks_3m:
                if d_3m[indices] != 0:
                    print(f"错误: 3m 张量元素 d{indices} 为 {d_3m[indices]}, 预期为: 0")
                    valid_3m = False
            
            if valid_3m:
                 print("成功: 3m 符号张量验证通过.")
            else:
                 print("失败: 3m 符号张量验证失败.")
        else:
            print(f"错误: 获取 {pg_3m} 的符号张量失败, 或形状不正确/全为零.")
        print("--- 完成测试点群 '3m = C₃ᵥ (trigonal)' 的符号张量 ---")

        print("\n--- 开始测试点群 '1 = C₁ (triclinic)' 的符号张量 --- ")
        pg_1 = "1 = C₁ (triclinic)"
        d_1 = get_symbolic_d_tensor(pg_1)
        if d_1 is not None and d_1.shape == (3,3,3) and d_1 != sympy.MutableDenseNDimArray.zeros(3,3,3):
            print(f"为 {pg_1} 创建张量成功.")
            all_symbols = set()
            is_correct_pg1 = True
            coord_map = ['x','y','z']
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        elem = d_1[i,j,k]
                        expected_sym_name = f"d_{coord_map[i]}{coord_map[j]}{coord_map[k]}"
                        if not isinstance(elem, sympy.Symbol) or str(elem) != expected_sym_name:
                            print(f"错误: 元素 d_{i}{j}{k} ({expected_sym_name}) 为 {elem}, 而非符号 {expected_sym_name}")
                            is_correct_pg1 = False
                            break
                        all_symbols.add(elem)
                if not is_correct_pg1: break
            
            if not is_correct_pg1:
                 print("失败: 点群 '1' 符号张量验证失败 (元素不符合预期).")
            elif len(all_symbols) == 27:
                print("成功: 点群 '1' 符号张量包含27个独立符号, 符合预期.")
            else:
                print(f"失败: 点群 '1' 符号张量包含 {len(all_symbols)} 个独立符号, 预期为27个.")
        else:
            print(f"错误: 获取 {pg_1} 的符号张量失败, 或形状不正确/全为零.")
        print("--- 完成测试点群 '1 = C₁ (triclinic)' 的符号张量 ---")
            
        # 确定 Oₕ 点群的正确关键字
        pg_Oh_actual_key = None
        # ₕ 是下标 h. point_group_data.json 使用这个 unicode 字符.
        oh_keys_to_check = ["m-3m = O\u2095 (cubic)", "m-3m = Oh (cubic)", "m-3m = O_h (cubic)"]
        for key_try in oh_keys_to_check:
            if key_try in POINT_GROUP_DATA:
                pg_Oh_actual_key = key_try
                print(f"通过直接匹配找到 Oₕ 点群关键字: '{pg_Oh_actual_key}'")
                break
        if not pg_Oh_actual_key:
             print("未通过直接匹配找到 Oₕ 关键字, 尝试通用搜索...")
             for k in POINT_GROUP_DATA.keys(): 
                if "m-3m" in k and ("O" in k.upper() and ("H" in k.upper() or "ₕ" in k)):
                    pg_Oh_actual_key = k
                    print(f"通过通用搜索找到 Oₕ 点群关键字: '{pg_Oh_actual_key}'")
                    break

        if pg_Oh_actual_key:
            print(f"\n--- 开始测试点群 '{pg_Oh_actual_key}' (中心对称群) 的SHG表达式 --- ")
            expressions_Oh = calculate_symbolic_shg_expressions(pg_Oh_actual_key)
            if expressions_Oh:
                all_expr_zero_Oh = all(expr == sympy.S.Zero for expr in expressions_Oh.values())
                is_non_zero_list_empty_in_json = (POINT_GROUP_DATA[pg_Oh_actual_key].get('Non-zero components', []) == [])
                
                if is_non_zero_list_empty_in_json and all_expr_zero_Oh:
                    print(f"成功: 中心对称群 {pg_Oh_actual_key} 的张量和SHG表达式均为零 (JSON中非零列表为空), 符合预期.")
                elif all_expr_zero_Oh:
                    print(f"成功: 群 {pg_Oh_actual_key} 的SHG表达式均为零.") # 也可能是正确的，即使非零列表不为空（例如，如果它们因对称性相互抵消为零）
                else:
                    print(f"失败: 群 {pg_Oh_actual_key} 的测试未通过. JSON非零列表空? {is_non_zero_list_empty_in_json}. 所有表达式为零? {all_expr_zero_Oh}.")
            else:
                print(f"错误: 计算 {pg_Oh_actual_key} 的SHG表达式失败 (返回 None 或空).")
            print(f"--- 完成测试点群 '{pg_Oh_actual_key}' 的SHG表达式 ---")
        else:
            print("\n--- 未找到 Oₕ (m-3m) 点群的关键字, 跳过其SHG表达式测试. ---")
            print(f"--- 完成测试点群 'Oₕ (m-3m) (未找到)' 的SHG表达式 ---")

        print("\n--- 开始测试点群 '3m = C₃ᵥ (trigonal)' 的SHG表达式 --- ")
        expressions_3m = calculate_symbolic_shg_expressions(pg_3m)
        if expressions_3m and expressions_3m['I_total'] != sympy.S.Zero:
            print(f"为 {pg_3m} 成功计算SHG表达式.")
            print(f"  Px 分量: {expressions_3m['Px']}")
            print(f"  Py 分量: {expressions_3m['Py']}")
            print(f"  Pz 分量: {expressions_3m['Pz']}")
            print(f"  总强度 (符号形式): {expressions_3m['I_total']}")
            print(f"  平行强度 (符号形式): {expressions_3m['I_parallel']}")
            print(f"  垂直强度 (符号形式): {expressions_3m['I_perpendicular']}")
        elif expressions_3m and expressions_3m['I_total'] == sympy.S.Zero:
            print(f"警告: 为 {pg_3m} 计算的SHG表达式中, 总强度为零. 这对于3m可能不符合预期.")
            for key, val_expr in expressions_3m.items(): print(f"  {key}: {val_expr}")
        else:
            print(f"错误: 计算 {pg_3m} 的SHG表达式失败.")
        print("--- 完成测试点群 '3m = C₃ᵥ (trigonal)' 的SHG表达式 ---")

    else:
        print("POINT_GROUP_DATA 数据为空. 无法执行测试.")
    print("\n所有测试执行完毕.") 