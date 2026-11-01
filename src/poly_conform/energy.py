"""Polymer Contact Energy and Metropolis Criterion for Coil-to-Globule Transitions."""

import numpy as np
from .conformation import PolymerChain


def count_nonbonded_contacts(chain: PolymerChain) -> int:
    """
    Count nearest-neighbor non-bonded contacts on 3D cubic lattice.
    Pairs (i, j) with |i - j| > 1 such that ||r_i - r_j||_1 == 1.
    """
    coords = chain.coords
    n = len(coords)
    site_map = {tuple(coords[i]): i for i in range(n)}

    # 6 nearest neighbors on cubic lattice
    directions = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ]

    contact_count = 0
    for i in range(n):
        x, y, z = coords[i]
        for dx, dy, dz in directions:
            neighbor = (x + dx, y + dy, z + dz)
            if neighbor in site_map:
                j = site_map[neighbor]
                # Avoid double counting and exclude adjacent chain bonds
                if j > i + 1:
                    contact_count += 1

    return contact_count


def compute_conformational_energy(chain: PolymerChain, epsilon: float = 1.0) -> float:
    """
    Conformational energy: E = - epsilon * N_contacts.
    Attractive contact energy epsilon > 0 drives collapse into a compact globule.
    """
    contacts = count_nonbonded_contacts(chain)
    return -epsilon * float(contacts)


def metropolis_criterion(delta_energy: float, temperature: float, rng: np.random.Generator) -> bool:
    """Standard Metropolis Monte Carlo acceptance rule."""
    if delta_energy <= 0.0:
        return True
    if temperature <= 0.0:
        return False
    prob = np.exp(-delta_energy / temperature)
    return rng.uniform(0.0, 1.0) < prob
