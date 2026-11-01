"""3D Lattice Polymer Conformation, Radius of Gyration, and Anisotropy."""

import numpy as np


class PolymerChain:
    """3D cubic lattice polymer conformation."""

    def __init__(self, coordinates: np.ndarray):
        coords = np.asarray(coordinates, dtype=np.int64)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("Coordinates must have shape (N, 3)")
        if len(coords) < 2:
            raise ValueError("Polymer chain must have at least 2 monomers")

        self.coords = coords
        self.length = len(coords)

    @classmethod
    def linear_chain(cls, length: int) -> "PolymerChain":
        """Initialize an extended linear rod along the x-axis."""
        coords = np.zeros((length, 3), dtype=np.int64)
        coords[:, 0] = np.arange(length)
        return cls(coords)

    def is_self_avoiding(self) -> bool:
        """Verify that no two monomers occupy the same lattice site."""
        unique_count = len(set(map(tuple, self.coords)))
        return unique_count == self.length

    def end_to_end_vector(self) -> np.ndarray:
        """Vector connecting first and last monomers: R_ee = r_N - r_0."""
        return self.coords[-1] - self.coords[0]

    def end_to_end_distance_sq(self) -> float:
        """Squared end-to-end distance: ||R_ee||^2."""
        v = self.end_to_end_vector()
        return float(np.sum(v * v))

    def center_of_mass(self) -> np.ndarray:
        """Center of mass: r_com = (1 / N) * sum(r_i)."""
        return np.mean(self.coords, axis=0, dtype=np.float64)

    def gyration_tensor(self) -> np.ndarray:
        """
        3x3 Radius of Gyration tensor:
        S_ab = (1 / N) * sum_i (r_ia - r_com_a)(r_ib - r_com_b)
        """
        r_com = self.center_of_mass()
        dr = self.coords.astype(np.float64) - r_com
        return (dr.T @ dr) / float(self.length)

    def radius_of_gyration_sq(self) -> float:
        """
        Squared radius of gyration: R_g^2 = Tr(S).
        Identical to (1 / (2 * N^2)) * sum_{i,j} ||r_i - r_j||^2.
        """
        s = self.gyration_tensor()
        return float(np.trace(s))

    def radius_of_gyration(self) -> float:
        """Root-mean-square radius of gyration R_g."""
        return float(np.sqrt(max(0.0, self.radius_of_gyration_sq())))

    def asphericity(self) -> float:
        """
        Asphericity parameter b from gyration tensor eigenvalues lambda_1 >= lambda_2 >= lambda_3:
        b = lambda_1 - 0.5 * (lambda_2 + lambda_3)
        Measures departure from spherical symmetry (0 for sphere, >0 for elongated/rod).
        """
        s = self.gyration_tensor()
        eigvals = np.sort(np.linalg.eigvalsh(s))[::-1]
        return float(eigvals[0] - 0.5 * (eigvals[1] + eigvals[2]))
