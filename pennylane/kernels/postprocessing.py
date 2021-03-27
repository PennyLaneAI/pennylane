# Copyright

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This file contains functionalities for postprocessing of kernel matrices.
"""
import numpy as np


def threshold_matrix(K):
    """Remove negative eigenvalues from the given kernel matrix.

    This method yields the closest positive semidefinite matrix in
    any unitarily invariant norm, e.g. the Frobenius norm.

    Args:
        K (array[float]): Kernel matrix assumed to be symmetric

    Returns:
        array[float]: Kernel matrix with negative eigenvalues cropped.
    """
    w, v = np.linalg.eigh(K)

    if np.min(w) < 0:
        w0 = np.clip(w, 0, None)

        return v @ np.diag(w0) @ np.transpose(v)

    return K


def displace_matrix(K):
    """Remove negative eigenvalues from the given kernel matrix by adding the identity matrix.

    This method has the advantage that it keeps the eigenvectors intact.

    Args:
        K (array[float]): Kernel matrix assumed to be symmetric

    Returns:
        array[float]: Kernel matrix with negative eigenvalues offset by adding the identity.
    """
    wmin = np.min(np.linalg.eigvalsh(K))

    if wmin < 0:
        return K - np.eye(K.shape[0]) * wmin

    return K


def closest_psd_matrix(K, fix_diagonal=True, solver=None):
    """Return the closest positive semidefinite matrix to the given kernel matrix.

    Args:
        K (array[float]): Kernel matrix assumed to be symmetric.
        fix_diagonal (bool): Whether to fix the diagonal to 1.
        solver (str, optional): Solver to be used by cvxpy. Defaults to CVXOPT.

    Returns:
        array[float]: closest positive semidefinite matrix in Frobenius norm.

    Comments:
        Requires cvxpy and the used solver (default CVXOPT) to be installed.
    """
    if not fix_diagonal:
        wmin = np.min(np.linalg.eigvalsh(K))
        if wmin >= 0:
            return K
    try:
        import cvxpy as cp

        if solver is None:
            solver = cp.CVXOPT
    except ImportError:
        # TODO: Make these proper warnings
        print("CVXPY is required for this post-processing method.", end="")
        if fix_diagonal:
            print(" As you want to fix the diagonal, task is impossible. Returning input...")
            return K
        else:
            print(" As you don't want to fix the diagonal, using threshold_matrix instead...")
            return threshold_matrix(K)

    X = cp.Variable(K.shape, PSD=True)
    constraint = [cp.diag(X) == 1.0] if fix_diagonal else []
    objective_fn = cp.norm(X - K, "fro")
    problem = cp.Problem(cp.Minimize(objective_fn), constraint)

    try:
        problem.solve(solver=solver)
    except Exception as e:
        problem.solve(verbose=True, solver=solver)

    return X.value


def mitigate_depolarization(K, num_wires, method, use_entries=None):
    """Estimate depolarizing noise rate(s) using on the diagonal entries of a kernel
    matrix and mitigate the noise, assuming a global depolarizing noise model.

    Args:
        K (array[float]): Noisy kernel matrix.
        num_wires (int): Number of wires/qubits of the quantum embedding kernel.
        method ('single'|'average'|'split_channel'): Strategy for mitigation
            'single': An alias for 'average' with len(use_entries)=1.
            'average': Estimate a globale noise rate based on the average of the diagonal
                entries in use_entries.
            'split_channel': Estimate individual noise rates per embedding.
        use_entries=None (array[int]): Diagonal entries to use if method in ['single', 'average'].
            If None, defaults to [0] ('single') or range(len(K)) ('average').
    Returns:
        K_bar (array[float]): Mitigated kernel matrix.
    Comments:
        If method=='average', diagonal entries use_entries have to be measured on the QC.
        If method=='split_channel', all diagonal entries are required.
    """
    dim = 2 ** num_wires

    if method == "single":
        if use_entries is None:
            use_entries = (0,)
        diagonal_element = K[use_entries[0], use_entries[0]]
        noise_rate = (1 - diagonal_element) * dim / (dim - 1)
        mitigated_matrix = (K - noise_rate / dim) / (1 - noise_rate)

    elif method == "average":
        if use_entries is None:
            diagonal_elements = np.diag(K)
        else:
            diagonal_elements = np.diag(K)[use_entries]
        noise_rates = (1 - diagonal_elements) * dim / (dim - 1)
        mean_noise_rate = np.mean(noise_rates)
        mitigated_matrix = (K - mean_noise_rate / dim) / (1 - mean_noise_rate)

    elif method == "split_channel":
        eff_noise_rates = np.clip((1 - np.diag(K)) * dim / (dim - 1), 0.0, 1.0)
        noise_rates = 1 - np.sqrt(1 - eff_noise_rates)
        inverse_noise = (
            -np.outer(noise_rates, noise_rates)
            + noise_rates.reshape((1, len(K)))
            + noise_rates.reshape((len(K), 1))
        )
        mitigated_matrix = (K - inverse_noise / dim) / (1 - inverse_noise)

    return mitigated_matrix
