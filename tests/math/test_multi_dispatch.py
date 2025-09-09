# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Assertion test for multi_dispatch function/decorator"""
# pylint: disable=unused-argument,no-value-for-parameter,too-few-public-methods,wrong-import-order
import autoray
import numpy as onp
import pytest
from autoray import numpy as anp

from pennylane import grad as qml_grad
from pennylane import math as fn
from pennylane import numpy as np

pytestmark = pytest.mark.all_interfaces

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

test_multi_dispatch_stack_data = [
    [[1.0, 0.0], [2.0, 3.0]],
    ([1.0, 0.0], [2.0, 3.0]),
    onp.array([[1.0, 0.0], [2.0, 3.0]]),
    anp.array([[1.0, 0.0], [2.0, 3.0]]),
    np.array([[1.0, 0.0], [2.0, 3.0]]),
    jnp.array([[1.0, 0.0], [2.0, 3.0]]),
]


@pytest.mark.gpu
@pytest.mark.parametrize("dev", ["cpu", "cuda"])
@pytest.mark.parametrize("func", [fn.array, fn.eye])
def test_array_cuda(func, dev):
    """Test that a new Torch tensor created with math.array/math.eye preserves
    the Torch device"""
    if not torch.cuda.is_available() and dev == "cuda":
        pytest.skip("A GPU would be required to run this test, but CUDA is not available.")

    original = torch.tensor(3, device=dev)
    new = func(2, like=original)
    assert isinstance(new, torch.Tensor)
    assert new.device == original.device


@pytest.mark.parametrize("x", test_multi_dispatch_stack_data)
def test_multi_dispatch_stack(x):
    """Test that the decorated autoray function stack can handle all inputs"""
    stack = fn.multi_dispatch(argnum=0, tensor_list=0)(autoray.numpy.stack)
    res = stack(x)
    assert fn.allequal(res, [[1.0, 0.0], [2.0, 3.0]])


@pytest.mark.parametrize("x", test_multi_dispatch_stack_data)
def test_multi_dispatch_decorate(x):
    """Test decorating a standard numpy function for PennyLane"""

    @fn.multi_dispatch(argnum=[0], tensor_list=[0])
    def tensordot(x, like, axes=None):
        return np.tensordot(x[0], x[1], axes=axes)

    assert fn.allequal(tensordot(x, axes=(0, 0)).numpy(), 2)


test_data0 = [
    (1, 2, 3),
    [1, 2, 3],
    onp.array([1, 2, 3]),
    anp.array([1, 2, 3]),
    np.array([1, 2, 3]),
    torch.tensor([1, 2, 3]),
    jnp.array([1, 2, 3]),
]

test_data = [(x, x) for x in test_data0]


@pytest.mark.parametrize("t1,t2", test_data)
def test_multi_dispatch_decorate_argnum_none(t1, t2):
    """Test decorating a standard numpy function for PennyLane, automatically dispatching all inputs by choosing argnum=None"""

    @fn.multi_dispatch(argnum=None, tensor_list=None)
    def tensordot(tensor1, tensor2, like, axes=None):
        return np.tensordot(tensor1, tensor2, axes=axes)

    assert fn.allequal(tensordot(t1, t2, axes=(0, 0)).numpy(), 14)


test_data_values = [
    [[1, 2, 3] for _ in range(5)],
    [(1, 2, 3) for _ in range(5)],
    [np.array([1, 2, 3]) for _ in range(5)],
    [onp.array([1, 2, 3]) for _ in range(5)],
    [anp.array([1, 2, 3]) for _ in range(5)],
    [torch.tensor([1, 2, 3]) for _ in range(5)],
    [jnp.array([1, 2, 3]) for _ in range(5)],
]


@pytest.mark.parametrize("values", test_data_values)
def test_multi_dispatch_decorate_non_dispatch(values):
    """Test decorating a custom function for PennyLane including a non-dispatchable parameter"""

    @fn.multi_dispatch(argnum=0, tensor_list=0)
    def custom_function(values, like, coefficient=10):
        """
        A dummy custom function that computes coeff :math:`c \\sum_i (v_i)^T v_i` where :math:`v_i` are vectors in ``values``
        and :math:`c` is a fixed ``coefficient``.
        values is a list of vectors
        like can force the interface (optional)
        """
        return coefficient * np.sum([fn.dot(v, v) for v in values])

    assert fn.allequal(custom_function(values), 700)


@pytest.mark.all_interfaces
def test_unwrap():
    """Test that unwrap converts lists to lists and interface variables to numpy."""
    params = [
        [torch.tensor(2)],
        [[3, 4], torch.tensor([5, 6])],
        [jnp.array(0.5), jnp.array([6, 7])],
        torch.tensor(0.5),
    ]

    out = fn.unwrap(params)

    assert out[0] == [2]
    assert out[1][0] == [3, 4]
    assert fn.allclose(out[1][1], np.array([5, 6]))
    assert fn.get_interface(out[1][1]) == "numpy"

    assert out[2][0] == 0.5
    assert fn.allclose(out[2][1], np.array([6, 7]))
    assert fn.get_interface(out[2][1]) == "numpy"

    assert out[3] == 0.5


@pytest.mark.parametrize(
    ("n", "t", "gamma_ref"),
    [
        (
            0.1,
            jnp.array([0.2, 0.3, 0.4]),
            jnp.array([0.87941963, 0.90835799, 0.92757383]),
        ),
        (
            0.1,
            np.array([0.2, 0.3, 0.4]),
            np.array([0.87941963, 0.90835799, 0.92757383]),
        ),
        (
            0.1,
            onp.array([0.2, 0.3, 0.4]),
            onp.array([0.87941963, 0.90835799, 0.92757383]),
        ),
    ],
)
def test_gammainc(n, t, gamma_ref):
    """Test that the lower incomplete Gamma function is computed correctly."""
    gamma = fn.gammainc(n, t)

    assert np.allclose(gamma, gamma_ref)


def test_dot_autograd():

    x = np.array([1.0, 2.0], requires_grad=False)
    y = np.array([2.0, 3.0], requires_grad=True)

    res = fn.dot(x, y)
    assert isinstance(res, np.tensor)
    assert res.requires_grad
    assert fn.allclose(res, 8)

    assert fn.allclose(qml_grad(fn.dot)(x, y), x)


def test_dot_autograd_with_scalar():

    x = np.array(1.0, requires_grad=False)
    y = np.array([2.0, 3.0], requires_grad=True)

    res = fn.dot(x, y)
    assert isinstance(res, np.tensor)
    assert res.requires_grad
    assert fn.allclose(res, [2.0, 3.0])

    res = fn.dot(y, x)
    assert isinstance(res, np.tensor)
    assert res.requires_grad
    assert fn.allclose(res, [2.0, 3.0])


def test_dot_torch_with_scalar():

    x = torch.tensor(1.0)
    y = torch.tensor([2.0, 3.0])

    res = fn.dot(x, y)
    assert isinstance(res, torch.Tensor)
    assert fn.allclose(res, [2.0, 3.0])

    res = fn.dot(y, x)
    assert isinstance(res, torch.Tensor)
    assert fn.allclose(res, [2.0, 3.0])


def test_kron():
    """Test the kronecker product function."""
    x = torch.tensor([[1, 2], [3, 4]])
    y = np.array([[0, 5], [6, 7]])

    res = fn.kron(x, y)
    expected = torch.tensor([[0, 5, 0, 10], [6, 7, 12, 14], [0, 15, 0, 20], [18, 21, 24, 28]])

    assert fn.allclose(res, expected)


class TestMatmul:
    @pytest.mark.torch
    def test_matmul_torch(self):
        m1 = torch.tensor([[1, 0], [0, 1]])
        m2 = [[1, 2], [3, 4]]
        assert fn.allequal(fn.matmul(m1, m2), m2)
        assert fn.allequal(fn.matmul(m2, m1), m2)
        assert fn.allequal(fn.matmul(m2, m2, like="torch"), np.matmul(m2, m2))
        assert fn.allequal(fn.matmul(m1, m1), m1)


class TestDetach:
    """Test the utility function detach."""

    def test_numpy(self):
        """Test that detach works with NumPy and does not do anything."""
        x = onp.array(0.3)
        detached_x = fn.detach(x)
        assert x is detached_x

    def test_autograd(self):
        """Test that detach works with Autograd."""
        import autograd

        x = np.array(0.3, requires_grad=True)
        assert fn.requires_grad(x) is True
        detached_x = fn.detach(x)
        assert fn.requires_grad(detached_x) is False
        with pytest.warns(UserWarning, match="Output seems independent"):
            jac = autograd.jacobian(fn.detach)(x)
        assert fn.isclose(jac, jac * 0.0)

    @pytest.mark.parametrize("use_jit", [True, False])
    def test_jax(self, use_jit):
        """Test that detach works with JAX."""

        x = jax.numpy.array(0.3)
        func = jax.jit(fn.detach, static_argnums=1) if use_jit else fn.detach
        jac = jax.jacobian(func)(x)
        assert jax.numpy.isclose(jac, 0.0)

    def test_torch(self):
        """Test that detach works with Torch."""

        x = torch.tensor(0.3, requires_grad=True)
        assert x.requires_grad is True
        detached_x = fn.detach(x)
        assert detached_x.requires_grad is False
        jac = torch.autograd.functional.jacobian(fn.detach, x)
        assert fn.isclose(jac, jac * 0.0)


@pytest.mark.all_interfaces
class TestNorm:
    mats_intrf_norm = (
        (np.array([0.5, -1, 2]), "numpy", np.array(2), {}),
        (np.array([[5, 6], [-2, 3]]), "numpy", np.array(11), {}),
        (torch.tensor([0.5, -1, 2]), "torch", torch.tensor(2), {}),
        (torch.tensor([[5.0, 6.0], [-2.0, 3.0]]), "torch", torch.tensor(11), {"axis": (0, 1)}),
        (jnp.array([0.5, -1, 2]), "jax", jnp.array(2), {}),
        (jnp.array([[5, 6], [-2, 3]]), "jax", jnp.array(11), {}),
    )

    @pytest.mark.parametrize("arr, expected_intrf, expected_norm, kwargs", mats_intrf_norm)
    def test_inf_norm(self, arr, expected_intrf, expected_norm, kwargs):
        """Test that inf norm is correct and works for each interface."""
        computed_norm = fn.norm(arr, ord=np.inf, **kwargs)
        assert np.allclose(computed_norm, expected_norm)
        assert fn.get_interface(computed_norm) == expected_intrf

    @pytest.mark.parametrize(
        "arr",
        [
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array(
                [
                    [[0.123, 0.456, 0.789], [-0.123, -0.456, -0.789]],
                    [[1.23, 4.56, 7.89], [-1.23, -4.56, -7.89]],
                ]
            ),
            np.array(
                [
                    [
                        [0.123 - 0.789j, 0.456 + 0.456j, 0.789 - 0.123j],
                        [-0.123 + 0.789j, -0.456 - 0.456j, -0.789 + 0.123j],
                    ],
                    [
                        [1.23 + 4.56j, 4.56 - 7.89j, 7.89 + 1.23j],
                        [-1.23 - 7.89j, -4.56 + 1.23j, -7.89 - 4.56j],
                    ],
                ]
            ),
        ],
    )
    def test_autograd_norm_gradient(self, arr):
        """Test that qml.math.norm has the correct gradient with autograd
        when the order and axis are not specified."""
        norm = fn.norm(arr)
        expected_norm = onp.linalg.norm(arr)
        assert np.isclose(norm, expected_norm)

        grad = qml_grad(fn.norm)(arr)
        expected_grad = (norm**-1) * arr.conj()
        assert fn.allclose(grad, expected_grad)


@pytest.mark.all_interfaces
class TestSVD:
    mat = [
        [-0.00707107 + 0.0j, 1.00707107 + 0.0j],
        [0.99292893 + 0.0j, -0.00707107 + 0.0j],
    ]

    mats_interface = (
        (
            onp.array(mat),
            "numpy",
        ),
        (
            torch.tensor(mat),
            "torch",
        ),
        (
            jnp.array(mat),
            "jax",
        ),
    )

    @pytest.mark.parametrize("mat, expected_intrf", mats_interface)
    @pytest.mark.parametrize(
        "expected_results",
        [
            (
                [
                    [
                        [0.92388093 + 0.0j, -0.38268036 + 0.0j],
                        [-0.38268048 + 0.0j, -0.9238808 + 0.0j],
                    ],
                    [1.0100001, 0.98999995],
                    [[-0.3826802 + 0.0j, 0.9238809 + 0.0j], [-0.9238809 + 0.0j, -0.3826802 + 0.0j]],
                ],
            ),
        ],
    )
    def test_svd_full(self, mat, expected_intrf, expected_results):
        """Test that svd is correct and works for each interface. Asking for the full decomposition"""
        results_svd = fn.svd(mat, compute_uv=True)
        for n in range(len(expected_results)):
            assert fn.get_interface(results_svd[n]) == expected_intrf
        if expected_intrf == "tensorflow":
            recovered_matrix = fn.matmul(
                fn.matmul(
                    results_svd[0],
                    fn.diag(np.array(results_svd[1], dtype="complex128"), like=expected_intrf),
                ),
                results_svd[2],
                like=expected_intrf,
            )
        else:
            recovered_matrix = fn.matmul(
                fn.matmul(
                    results_svd[0],
                    fn.diag(results_svd[1], like=expected_intrf),
                ),
                results_svd[2],
                like=expected_intrf,
            )

        assert np.allclose(mat, recovered_matrix, rtol=1e-04)

    @pytest.mark.parametrize("mat, expected_intrf", mats_interface)
    @pytest.mark.parametrize(
        "expected_results",
        [
            ([[1.0100001, 0.98999995]]),
        ],
    )
    def test_svd_only_sv(self, mat, expected_intrf, expected_results):
        """Test that svd is correct and works for each interface. Asking only for singular values."""
        results_svd = fn.svd(mat, compute_uv=False)

        assert np.allclose(results_svd, expected_results)
        assert fn.get_interface(results_svd) == expected_intrf
