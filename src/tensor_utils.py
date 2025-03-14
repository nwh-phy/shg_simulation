import numpy as np

def str_to_indices(comp_str):
    """Convert 'xyz' type string to a tuple of integer indices."""
    mapping = {'x': 0, 'y': 1, 'z': 2}
    return tuple(mapping[c] for c in comp_str.lower())

def create_tensor(components, dim=3, scale=1.0):
    """Create a tensor based on string components and scale."""
    tensor = np.zeros((dim, dim, dim))
    for comp_str in components:
        indices = str_to_indices(comp_str)
        tensor[indices] = scale  # Scale the tensor component
    return tensor

def scale_tensor(tensor, scale_factor):
    """Scale the tensor by a given factor."""
    return tensor * scale_factor

def tensor_norm(tensor):
    """Calculate the norm of the tensor."""
    return np.linalg.norm(tensor)