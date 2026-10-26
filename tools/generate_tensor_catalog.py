"""Derive tensors from Neumann invariance and SHG j/k symmetry, without Kleinman.

Cartesian axes: unique b for monoclinic 2; z principal axis for 3/4/6.
3m and -6m2 mirror yz. Regenerate: python tools/generate_tensor_catalog.py
"""
import json
from pathlib import Path
import sympy as s


def generate():
    path = Path(__file__).resolve().parents[1] / "data" / "point_group_data.json"
    names = list(json.loads(path.read_text(encoding="utf-8"))["point_groups"])
    corrections = {'4̄2m = D₂ₘ (tetragonal)': '4̄2m = D₂d (tetragonal)',
                   '4̄3m = Tₘ (cubic)': '4̄3m = Td (cubic)',
                   '3̄m = D₃ₘ (trigonal)': '3̄m = D₃d (trigonal)',
                   'm3 = Tₕ (cubic)': 'm3̄ = Tₕ (cubic)', 'm3m = Oₕ (cubic)': 'm3̄m = Oₕ (cubic)'}
    names = [corrections.get(name, name) for name in names]
    inv = -s.eye(3)
    cx, cy = s.diag(1, -1, -1), s.diag(-1, 1, -1)
    my = s.diag(-1, 1, 1)
    r4 = s.Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    r3 = s.Matrix([[-s.Rational(1, 2), -s.sqrt(3)/2, 0], [s.sqrt(3)/2, -s.Rational(1, 2), 0], [0, 0, 1]])
    r6 = s.Matrix([[s.Rational(1, 2), -s.sqrt(3)/2, 0], [s.sqrt(3)/2, s.Rational(1, 2), 0], [0, 0, 1]])
    cycle = s.Matrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    generators = [[], [inv], [cy], [s.diag(1, -1, 1)], [cy, inv], [cx, cy], [cx*cy, my], [cx, cy, inv],
                  [r4], [-r4], [r4, cx], [r4, my], [-r4, cx], [r4, inv], [r4, cx, inv],
                  [r4, cycle], [-r4, cycle], [cx, cycle], [cx, cycle, inv], [r4, cycle, inv],
                  [r3], [r3, cx], [r3, my], [-r3], [r3, cx, inv],
                  [r6], [-r6], [r6, cx], [r6, my], [-r6, cy], [r6, inv], [r6, cx, inv]]
    symbols = ['1', '1̄', '2', 'm', '2/m', '222', 'mm2', 'mmm', '4', '4̄', '422', '4mm', '4̄2m', '4/m', '4/mmm', '432', '4̄3m', '23', 'm3̄', 'm3̄m', '3', '32', '3m', '3̄', '3̄m', '6', '6̄', '622', '6mm', '6̄m2', '6/m', '6/mmm']
    generator_map = dict(zip(symbols, generators))
    if {name.split(' =')[0] for name in names} != set(symbols):
        raise ValueError('The catalog must contain the supported 32 point groups.')
    pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    coordinates = [(i, j, k) for i in range(3) for j, k in pairs]
    embed, select = s.zeros(27, 18), s.zeros(18, 27)
    for col, (i, j, k) in enumerate(coordinates):
        embed[i*9+j*3+k, col] = embed[i*9+k*3+j, col] = 1
        select[col, i*9+j*3+k] = 1
    result = {}
    for name in names:
        operations = generator_map[name.split(' =')[0]]
        constraints = s.zeros(0, 18)
        for g in operations:
            constraints = constraints.col_join(select*s.kronecker_product(g, g, g)*embed - s.eye(18))
        vectors = constraints.nullspace() if operations else [s.eye(18)[:, i] for i in range(18)]
        basis = {}
        for vector in vectors:
            free = max(i for i, value in enumerate(vector) if value != 0)
            label = ''.join('xyz'[i] for i in coordinates[free])
            full = embed*vector
            basis[label] = {''.join('xyz'[i] for i in (n//9, n//3 % 3, n % 3)): int(value) if value.is_Integer else float(value) for n, value in enumerate(full) if value != 0}
        nonzero = sorted({c for values in basis.values() for c in values})
        relations = []
        for param, values in basis.items():
            terms = [param] + [('' if value == 1 else '-') + comp for comp, value in values.items() if comp != param and value in (-1, 1)]
            if len(terms) > 1:
                relations.append(' = '.join(terms))
        result[name] = {"Non-zero components": nonzero, "Independent components": list(basis), "Relations": relations, "Basis": basis,
                        "Generators": [[[float(x) for x in row] for row in g.tolist()] for g in operations]}
        print(name, len(basis))
    path.write_text(json.dumps({"schema_version": 2, "convention": "Relative d_ijk=chi_ijk/2; jk symmetric; no Kleinman assumption. P/(2 epsilon0)=d:E:E. Active ZYZ Rz(phi)Ry(theta)Rz(psi). 3m/-6m2 mirror yz; monoclinic unique axis y.", "derivation": "tools/generate_tensor_catalog.py: exact nullspace of point-group invariance with SHG jk symmetry", "references": ["https://bfichera.github.io/shgpy/tutorial/tensor_tutorial.html", "https://doi.org/10.1038/s41524-022-00930-4"], "point_groups": result}, ensure_ascii=False, indent=2) + '\n', encoding="utf-8")


if __name__ == "__main__":
    generate()
