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

    if fix_diagonal:
        constraint = [cp.diag(X) == 1.]
    else:
        wmin = np.min(np.linalg.eigvalsh(K))
        if wmin >= 0:
            return K
        constraint = []

    X = cp.Variable(K.shape, PSD=True)
    objective_fn = cp.norm(X - K, "fro")
    problem = cp.Problem(cp.Minimize(objective_fn), constraint)

    try:
        problem.solve(solver=solver)
    except Exception as e:
        problem.solve(verbose=True, solver=solver)

    return X.value
