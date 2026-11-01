"""Monte Carlo Equilibrium Ensemble Sampler for Lattice Polymers."""

import numpy as np
from typing import Dict, Any
from .conformation import PolymerChain
from .energy import compute_conformational_energy, count_nonbonded_contacts, metropolis_criterion
from .pivot import pivot_step


def sample_polymer_ensemble(
    length: int,
    total_steps: int = 1000,
    burn_in: int = 200,
    epsilon: float = 0.0,
    temperature: float = 1.0,
    sample_interval: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run Pivot Monte Carlo sampling on a polymer of given length.
    Returns dictionary with time series and ensemble averages.
    """
    rng = np.random.default_rng(seed)
    current_chain = PolymerChain.linear_chain(length)
    current_energy = compute_conformational_energy(current_chain, epsilon)

    rg_values = []
    ree_values = []
    energy_values = []
    contacts_values = []
    accepted_moves = 0
    attempted_moves = 0

    for step in range(total_steps):
        # Propose geometric pivot move
        candidate, valid_saw = pivot_step(current_chain, rng)
        attempted_moves += 1

        if valid_saw:
            # Energetic Metropolis check (if epsilon != 0)
            if epsilon != 0.0:
                cand_energy = compute_conformational_energy(candidate, epsilon)
                delta_e = cand_energy - current_energy
                if metropolis_criterion(delta_e, temperature, rng):
                    current_chain = candidate
                    current_energy = cand_energy
                    accepted_moves += 1
            else:
                current_chain = candidate
                accepted_moves += 1

        # Record samples post burn-in
        if step >= burn_in and step % sample_interval == 0:
            rg = current_chain.radius_of_gyration()
            ree = np.sqrt(current_chain.end_to_end_distance_sq())
            contacts = count_nonbonded_contacts(current_chain)

            rg_values.append(rg)
            ree_values.append(ree)
            energy_values.append(current_energy)
            contacts_values.append(contacts)

    rg_arr = np.array(rg_values, dtype=np.float64)
    ree_arr = np.array(ree_values, dtype=np.float64)

    return {
        "mean_rg": float(np.mean(rg_arr)) if len(rg_arr) > 0 else 0.0,
        "mean_rg_sq": float(np.mean(rg_arr ** 2)) if len(rg_arr) > 0 else 0.0,
        "mean_ree": float(np.mean(ree_arr)) if len(ree_arr) > 0 else 0.0,
        "mean_ree_sq": float(np.mean(ree_arr ** 2)) if len(ree_arr) > 0 else 0.0,
        "mean_contacts": float(np.mean(contacts_values)) if len(contacts_values) > 0 else 0.0,
        "acceptance_ratio": float(accepted_moves / max(1, attempted_moves)),
        "samples_count": len(rg_values),
    }
