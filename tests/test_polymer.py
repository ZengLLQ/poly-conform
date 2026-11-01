import numpy as np
import pytest
from poly_conform import (
    PolymerChain,
    count_nonbonded_contacts,
    compute_conformational_energy,
    pivot_step,
    generate_cubic_symmetries,
    sample_polymer_ensemble,
    estimate_flory_exponent,
)


def test_linear_chain_analytical_radius_of_gyration():
    # For a continuous linear rod of length N discrete beads along an axis:
    # Rg^2 = (N^2 - 1) / 12
    n = 7
    chain = PolymerChain.linear_chain(n)

    expected_rg_sq = (n**2 - 1) / 12.0  # 48 / 12 = 4.0
    expected_rg = 2.0

    assert np.isclose(chain.radius_of_gyration_sq(), expected_rg_sq, atol=1e-12)
    assert np.isclose(chain.radius_of_gyration(), expected_rg, atol=1e-12)
    assert chain.is_self_avoiding()

    # End-to-end distance of linear chain: N - 1
    assert np.isclose(chain.end_to_end_distance_sq(), (n - 1) ** 2)

    # Gyration tensor trace must equal Rg^2
    s_mat = chain.gyration_tensor()
    assert np.isclose(np.trace(s_mat), expected_rg_sq, atol=1e-12)

    # Asphericity of 1D rod is strictly positive
    assert chain.asphericity() > 0.0


def test_self_avoiding_collision_detection():
    # Create a self-colliding chain
    colliding_coords = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 0], # Collision with monomer 0
    ])

    chain = PolymerChain(colliding_coords)
    assert not chain.is_self_avoiding()


def test_cubic_symmetries_group_order():
    syms = generate_cubic_symmetries()
    # Octahedral symmetry group O_h has exactly 48 elements
    assert len(syms) == 48

    identity = np.eye(3, dtype=np.int64)
    for m in syms:
        # Must be orthogonal: M * M^T = I
        assert np.array_equal(m @ m.T, identity)
        det = int(np.round(np.linalg.det(m)))
        assert det in (-1, 1)


def test_pivot_move_preserves_saw_invariance():
    rng = np.random.default_rng(42)
    chain = PolymerChain.linear_chain(20)

    # Perform 50 pivot moves
    for _ in range(50):
        new_chain, accepted = pivot_step(chain, rng)
        if accepted:
            assert new_chain.is_self_avoiding()
            chain = new_chain

    # Chain should have coiled (end-to-end distance smaller than linear 19)
    assert np.sqrt(chain.end_to_end_distance_sq()) < 19.0


def test_nonbonded_contact_counting():
    # Linear chain has 0 non-bonded contacts
    linear = PolymerChain.linear_chain(10)
    assert count_nonbonded_contacts(linear) == 0
    assert compute_conformational_energy(linear, epsilon=1.5) == 0.0

    # 4-monomer U-turn loop:
    # 0: (0, 0, 0)
    # 1: (1, 0, 0)
    # 2: (1, 1, 0)
    # 3: (0, 1, 0)
    # Monomers 0 and 3 are separated by distance 1, and |3 - 0| = 3 > 1 -> 1 contact
    u_loop = PolymerChain(np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
    ]))
    assert count_nonbonded_contacts(u_loop) == 1
    assert compute_conformational_energy(u_loop, epsilon=2.0) == -2.0


def test_ensemble_sampler_and_flory_scaling():
    # Sample SAW ensembles for lengths 10, 16, 24
    lengths = [10, 16, 24]
    rg_sq_means = []

    for length in lengths:
        res = sample_polymer_ensemble(
            length=length,
            total_steps=500,
            burn_in=100,
            epsilon=0.0,
            sample_interval=5,
            seed=42,
        )
        assert res["samples_count"] > 0
        assert res["mean_rg"] > 0.0
        rg_sq_means.append(res["mean_rg_sq"])

    # Fit Flory exponent: <Rg^2> ~ N^(2 * nu)
    nu, r_sq = estimate_flory_exponent(lengths, rg_sq_means)

    # 3D Flory exponent for self-avoiding walks is ~0.588
    # For short chains on lattice, nu typically falls in 0.45 - 0.70
    assert 0.40 < nu < 0.75
    assert r_sq > 0.90
