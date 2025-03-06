# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
    r"""Remove negative eigenvalues from the given kernel matrix.

    This method yields the closest positive semi-definite matrix in
    any unitarily invariant norm, e.g. the Frobenius norm.

    Args:
        K (array[float]): Kernel matrix, assumed to be symmetric.

    Returns:
        array[float]: Kernel matrix with cropped negative eigenvalues.

    **Example:**

    Consider a symmetric matrix with both positive and negative eigenvalues:

    .. code-block :: pycon

        >>> K = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 2]])
        >>> np.linalg.eigvalsh(K)
        array([-1.,  1.,  2.])

    We then can threshold/truncate the eigenvalues of the matrix via

    .. code-block :: pycon

        >>> K_thresh = qml.kernels.threshold_matrix(K)
        >>> np.linalg.eigvalsh(K_thresh)
        array([0.,  1.,  2.])

    If the input matrix does not have negative eigenvalues, ``threshold_matrix``
    does not have any effect.
    """
    w, v = np.linalg.eigh(K)

    if w[0] < 0:
        # Transform spectrum: Threshold/clip at 0.
        w0 = np.clip(w, 0, None)

        return (v * w0) @ np.transpose(v)

    return K


def displace_matrix(K):
    r"""Remove negative eigenvalues from the given kernel matrix by adding a multiple
    of the identity matrix.

    This method keeps the eigenvectors of the matrix intact.

    Args:
        K (array[float]): Kernel matrix, assumed to be symmetric.

    Returns:
        array[float]: Kernel matrix with eigenvalues offset by adding the identity.

    **Example:**

    Consider a symmetric matrix with both positive and negative eigenvalues:

    .. code-block :: pycon

        >>> K = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 2]])
        >>> np.linalg.eigvalsh(K)
        array([-1.,  1.,  2.])

    We then can shift all eigenvalues of the matrix by adding the identity matrix
    multiplied with the absolute value of the smallest (the most negative, that is)
    eigenvalue:

    .. code-block :: pycon

        >>> K_displaced = qml.kernels.displace_matrix(K)
        >>> np.linalg.eigvalsh(K_displaced)
        array([0.,  2.,  3.])

    If the input matrix does not have negative eigenvalues, ``displace_matrix``
    does not have any effect.
    """
    wmin = np.linalg.eigvalsh(K)[0]

    if wmin < 0:
        return K - np.eye(K.shape[0]) * wmin

    return K


def flip_matrix(K):
    r"""Remove negative eigenvalues from the given kernel matrix by taking the absolute value.

    This method keeps the eigenvectors of the matrix intact.

    Args:
        K (array[float]): Kernel matrix, assumed to be symmetric.

    Returns:
        array[float]: Kernel matrix with flipped negative eigenvalues.

    Reference:
        This method is introduced in
        `Wang, Du, Luo & Tao (2021) <https://doi.org/10.22331/q-2021-08-30-531>`_.

    **Example:**

    Consider a symmetric matrix with both positive and negative eigenvalues:

    .. code-block :: pycon

        >>> K = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 2]])
        >>> np.linalg.eigvalsh(K)
        array([-1.,  1.,  2.])

    We then can invert the sign of all negative eigenvalues of the matrix, obtaining
    non-negative eigenvalues only:

    .. code-block :: pycon

        >>> K_flipped = qml.kernels.flip_matrix(K)
        >>> np.linalg.eigvalsh(K_flipped)
        array([1.,  1.,  2.])

    If the input matrix does not have negative eigenvalues, ``flip_matrix``
    does not have any effect.
    """
    w, v = np.linalg.eigh(K)

    if w[0] < 0:
        # Transform spectrum: absolute value
        w_abs = np.abs(w)

        return (v * w_abs) @ np.transpose(v)

    return K


def closest_psd_matrix(K, fix_diagonal=False, solver=None, **kwargs):
    r"""Return the closest positive semi-definite matrix to the given kernel matrix.

    This method either fixes the diagonal entries to be 1
    (``fix_diagonal=True``) or keeps the eigenvectors intact (``fix_diagonal=False``),
    in which case it reduces to the method :func:`~.kernels.threshold_matrix`.
    For ``fix_diagonal=True`` a semi-definite program is solved.

    Args:
        K (array[float]): Kernel matrix, assumed to be symmetric.
        fix_diagonal (bool): Whether to fix the diagonal to 1.
        solver (str, optional): Solver to be used by cvxpy. Defaults to CVXOPT.
        kwargs (kwarg dict): Passed to cvxpy.Problem.solve().

    Returns:
        array[float]: closest positive semi-definite matrix in Frobenius norm.

    Comments:
        Requires cvxpy and the used solver (default CVXOPT) to be installed if ``fix_diagonal=True``.

    Reference:
        This method is introduced in `arXiv:2105.02276 <https://arxiv.org/abs/2105.02276>`_.

    **Example:**

    Consider a symmetric matrix with both positive and negative eigenvalues:

    .. code-block :: pycon

        >>> K = np.array([[0.9, 1.], [1., 0.9]])
        >>> np.linalg.eigvalsh(K)
        array([-0.1, 1.9])

    The positive semi-definite matrix that is closest to this matrix in any unitarily
    invariant norm is then given by the matrix with the eigenvalues thresholded at 0,
    as computed by :func:`~.kernels.threshold_matrix`:

    .. code-block :: pycon

        >>> K_psd = qml.kernels.closest_psd_matrix(K)
        >>> K_psd
        array([[0.95, 0.95],
               [0.95, 0.95]])
        >>> np.linalg.eigvalsh(K_psd)
        array([0., 1.9])
        >>> np.allclose(K_psd, qml.kernels.threshold_matrix(K))
        True

    However, for quantum kernel matrices we may want to restore the value 1 on the
    diagonal:

    .. code-block :: pycon

        >>> K_psd = qml.kernels.closest_psd_matrix(K, fix_diagonal=True)
        >>> K_psd
        array([[1.        , 0.99998008],
               [0.99998008, 1.        ]])
        >>> np.linalg.eigvalsh(K_psd)
        array([1.99162415e-05, 1.99998008e+00])

    If the input matrix does not have negative eigenvalues and ``fix_diagonal=False``,
    ``closest_psd_matrix`` does not have any effect.
    """
    if not fix_diagonal:
        return threshold_matrix(K)
    try:
        import cvxpy as cp  # pylint: disable=import-outside-toplevel

        if solver is None:
            solver = cp.CVXOPT
    except ImportError as e:
        raise ImportError("CVXPY is required for this post-processing method.") from e

    X = cp.Variable(K.shape, PSD=True)
    constraint = [cp.diag(X) == 1.0] if fix_diagonal else []
    objective_fn = cp.norm(X - K, "fro")
    problem = cp.Problem(cp.Minimize(objective_fn), constraint)

    try:
        problem.solve(solver=solver, **kwargs)
    except Exception:  # pylint: disable=broad-except
        try:
            problem.solve(solver=solver, verbose=True, **kwargs)
        except Exception as e:
            raise RuntimeError("CVXPY solver did not converge.") from e

    return X.value


def mitigate_depolarizing_noise(K, num_wires, method, use_entries=None):
    r"""Estimate depolarizing noise rate(s) using on the diagonal entries of a kernel
    matrix and mitigate the noise, assuming a global depolarizing noise model.

    Args:
        K (array[float]): Noisy kernel matrix.
        num_wires (int): Number of wires/qubits of the quantum embedding kernel.
        method (``'single'`` | ``'average'`` | ``'split_channel'``): Strategy for mitigation

            * ``'single'``: An alias for ``'average'`` with ``len(use_entries)=1``.
            * ``'average'``: Estimate a global noise rate based on the average of the diagonal
              entries in ``use_entries``, which need to be measured on the quantum computer.
            * ``'split_channel'``: Estimate individual noise rates per embedding, requiring
              all diagonal entries to be measured on the quantum computer.
        use_entries (array[int]): Diagonal entries to use if method in ``['single', 'average']``.
            If ``None``, defaults to ``[0]`` (``'single'``) or ``range(len(K))`` (``'average'``).

    Returns:
        array[float]: Mitigated kernel matrix.

    Reference:
        This method is introduced in Section V in
        `arXiv:2105.02276 <https://arxiv.org/abs/2105.02276>`_.

    **Example:**

    For an example usage of ``mitigate_depolarizing_noise`` please refer to the
    `PennyLane demo on the kernel module <https://github.com/PennyLaneAI/qml/tree/master/demonstrations/tutorial_kernel_based_training.py>`_ or `the postprocessing demo for arXiv:2105.02276 <https://github.com/thubregtsen/qhack/blob/master/paper/post_processing_demo.py>`_.
    """
    dim = 2**num_wires

    if method == "single":
        if use_entries is None:
            use_entries = (0,)

        if K[use_entries[0], use_entries[0]] <= (1 / dim):
            raise ValueError(
                "The single noise mitigation method cannot be applied "
                "as the single diagonal element specified is too small."
            )

        diagonal_element = K[use_entries[0], use_entries[0]]
        noise_rate = (1 - diagonal_element) * dim / (dim - 1)
        mitigated_matrix = (K - noise_rate / dim) / (1 - noise_rate)

    elif method == "average":
        if use_entries is None:
            diagonal_elements = np.diag(K)
        else:
            diagonal_elements = np.diag(K)[np.array(use_entries)]

        if np.mean(diagonal_elements) <= 1 / dim:
            raise ValueError(
                "The average noise mitigation method cannot be applied "
                "as the average of the used diagonal terms is too small."
            )

        noise_rates = (1 - diagonal_elements) * dim / (dim - 1)
        mean_noise_rate = np.mean(noise_rates)
        mitigated_matrix = (K - mean_noise_rate / dim) / (1 - mean_noise_rate)

    elif method == "split_channel":
        if np.any(np.diag(K) <= 1 / dim):
            raise ValueError(
                "The split channel noise mitigation method cannot be applied "
                "to the input matrix as its diagonal terms are too small."
            )

        eff_noise_rates = np.clip((1 - np.diag(K)) * dim / (dim - 1), 0.0, 1.0)
        noise_rates = 1 - np.sqrt(1 - eff_noise_rates)
        inverse_noise = (
            -np.outer(noise_rates, noise_rates)
            + noise_rates.reshape((1, len(K)))
            + noise_rates.reshape((len(K), 1))
        )

        mitigated_matrix = (K - inverse_noise / dim) / (1 - inverse_noise)
    else:
        raise ValueError(
            "Incorrect noise depolarization mitigation method specified. "
            "Accepted strategies are: 'single', 'average' and 'split_channel'."
        )

    return mitigated_matrix
