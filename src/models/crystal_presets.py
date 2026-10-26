"""Common symmetry templates; coefficients are dimensionless examples, not material constants."""
from point_groups import get_components_for_group

COMMON_CRYSTALS = {
    name: {"point_group": group, "coeffs": {c: 1.0 for c in get_components_for_group(group)},
           "description": "Symmetry example only; dimensionless, not a literature material coefficient"}
    for name, group in [("LiNbO3", "3m = C₃ᵥ (trigonal)"), ("KDP", "4̄2m = D₂d (tetragonal)")]
}
