"""Single source for symmetry-constrained SHG tensors; no Kleinman approximation."""
import json
from pathlib import Path
import sys
import numpy as np


def get_data_file_path(filename):
    root = Path(sys._MEIPASS) if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
    return str(root / "data" / filename)


def load_point_group_data(filename):
    return json.loads(Path(filename).read_text(encoding="utf-8-sig"))["point_groups"]


point_group_components = load_point_group_data(get_data_file_path("point_group_data.json"))


def str_to_indices(component):
    if len(component) != 3 or any(c not in "xyz" for c in component):
        raise ValueError(f"Invalid tensor component: {component}")
    return tuple("xyz".index(c) for c in component)


def get_all_point_groups():
    return list(point_group_components)


def get_components_for_group(group_name):
    """Only independent parameters are editable."""
    return list(point_group_components[group_name]["Independent components"])


def get_relations_for_group(group_name):
    return list(point_group_components[group_name]["Relations"])


def get_tensor_basis(group_name):
    basis = {}
    for label, entries in point_group_components[group_name]["Basis"].items():
        tensor = np.zeros((3, 3, 3))
        for component, coefficient in entries.items():
            tensor[str_to_indices(component)] = coefficient
        basis[label] = tensor
    return basis


def create_tensor_with_relations(group_name, dim=3, values=None):
    if dim != 3:
        raise ValueError("SHG tensor must have shape 3x3x3.")
    basis = get_tensor_basis(group_name)
    values = {p: 1.0 for p in basis} if values is None else values
    if set(values) - set(basis):
        raise ValueError("Unknown independent tensor parameter.")
    tensor = np.zeros((3, 3, 3), dtype=complex)
    for label, block in basis.items():
        value = complex(values.get(label, 0))
        if not np.isfinite(value):
            raise ValueError("Tensor parameters must be finite.")
        tensor += value * block
    return np.real_if_close(tensor)


def create_tensor(components, dim=3):
    tensor = np.zeros((dim, dim, dim))
    for component in components:
        i, j, k = str_to_indices(component)
        tensor[i, j, k] = tensor[i, k, j] = 1
    return tensor


def apply_relations(tensor, relations, dim=3):
    result = tensor.copy()
    for relation in relations:
        terms = [term.strip() for term in relation.split('=')]
        value = result[str_to_indices(terms[0].lstrip('-'))] * (-1 if terms[0].startswith('-') else 1)
        for term in terms:
            i, j, k = str_to_indices(term.lstrip('-'))
            result[i, j, k] = result[i, k, j] = value * (-1 if term.startswith('-') else 1)
    return (result + result.swapaxes(1, 2)) / 2


def show_component_table(group_name):
    print(group_name, get_components_for_group(group_name), get_relations_for_group(group_name))
