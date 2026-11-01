"""Madras-Sokal Pivot Algorithm for 3D Lattice Self-Avoiding Walk Conformation Sampling."""

import itertools
import numpy as np
from typing import List, Tuple
from .conformation import PolymerChain


def generate_cubic_symmetries() -> List[np.ndarray]:
    """Generate all 48 symmetry matrices of the 3D cubic point group O_h."""
    matrices = []
    base_permutations = list(itertools.permutations([0, 1, 2]))
    sign_combinations = list(itertools.product([-1, 1], repeat=3))

    for p in base_permutations:
        for signs in sign_combinations:
            m = np.zeros((3, 3), dtype=np.int64)
            for row in range(3):
                col = p[row]
                m[row, col] = signs[row]
            matrices.append(m)

    return matrices


CUBIC_SYMMETRIES = generate_cubic_symmetries()


def pivot_step(chain: PolymerChain, rng: np.random.Generator = None) -> Tuple[PolymerChain, bool]:
    """
    Attempt a single Madras-Sokal pivot step on the polymer chain.
    Returns: (new_chain, accepted)
    """
    if rng is None:
        rng = np.random.default_rng()

    n = chain.length
    if n <= 2:
        return chain, False

    # Pick random pivot site k in [0, n - 2]
    k = rng.integers(0, n - 1)
    pivot_pt = chain.coords[k]

    # Pick random symmetry operation R from O_h (excluding identity)
    sym_idx = rng.integers(0, len(CUBIC_SYMMETRIES))
    r_mat = CUBIC_SYMMETRIES[sym_idx]

    # Create candidate coordinates
    new_coords = chain.coords.copy()
    tail = chain.coords[k + 1:] - pivot_pt
    new_tail = (tail @ r_mat.T) + pivot_pt
    new_coords[k + 1:] = new_tail

    # Check self-avoidance (collision check)
    head_set = set(map(tuple, new_coords[:k + 1]))
    tail_set = set(map(tuple, new_coords[k + 1:]))

    # Tail must have no internal self-intersections and no overlap with head
    if len(tail_set) != len(new_coords[k + 1:]):
        return chain, False

    if not head_set.isdisjoint(tail_set):
        return chain, False

    return PolymerChain(new_coords), True
