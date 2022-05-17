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
"""
Unit tests for the :func:`pennylane.math.is_independent` function.
"""
import pytest

import numpy as np
from pennylane import numpy as pnp

import pennylane as qml
from pennylane.math import is_independent
from pennylane.math.is_independent import _get_random_args

pytestmark = pytest.mark.all_interfaces

tf = pytest.importorskip("tensorflow", minversion="2.1")
torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


dependent_lambdas = [
    lambda x: x,
    lambda x: (x, x),
    lambda x: [x] * 10,
    lambda x: (2.0 * x, x),
    lambda x: 0.0 * x,
    lambda x, y: (0.0 * x, 0.0 * y),
    lambda x: x if x > 0 else 0.0,  # RELU for x>0 is okay numerically
    lambda x: x if x > 0 else 0.0,  # RELU for x<0 is okay numerically
    lambda x: 1.0 if abs(x) < 1e-5 else 0.0,  # delta for x=0 is okay numerically
    lambda x: x if abs(x) < 1e-5 else 0.0,  # x*delta for x=0 is okay
    lambda x: 1.0 if x > 0 else 0.0,  # Heaviside is okay numerically
    lambda x: 1.0 if x > 0 else 0.0,  # Heaviside is okay numerically
    lambda x: qml.math.log(1 + qml.math.exp(100.0 * x)) / 100.0,  # Softplus is okay
    lambda x: qml.math.log(1 + qml.math.exp(100.0 * x)) / 100.0,  # Softplus is okay
]

args_dependent_lambdas = [
    (np.array(1.2),),
    (2.19,),
    (2.19,),
    (1.0,),
    (np.ones((2, 3)),),
    (np.array([2.0, 5.0]), 1.2),
    (1.6,),
    (-2.0,),
    (0.0,),
    (0.0,),
    (-2.0,),
    (2.0,),
    (-0.2,),
    (0.9,),
]

lambdas_expect_torch_fail = [
    False,
    False,
    False,
    False,
    True,
    True,
    False,
    False,
    False,
    True,
    False,
    False,
    False,
    False,
]

overlooked_lambdas = [
    lambda x: 1.0 if abs(x) < 1e-5 else 0.0,  # delta for x!=0 is not okay
    lambda x: 1.0 if abs(x) < 1e-5 else 0.0,  # delta for x!=0 is not okay
]

args_overlooked_lambdas = [
    (2.0,),
    (-2.0,),
]


class TestIsIndependentAutograd:
    """Tests for is_independent, which tests a function to be
    independent of its inputs, using Autograd."""

    interface = "autograd"

    @pytest.mark.parametrize("num", [0, 1, 2])
    @pytest.mark.parametrize(
        "args",
        [
            (0.2,),
            (1.1, 3.2, 0.2),
            (np.array([[0, 9.2], [-1.2, 3.2]]),),
            (0.3, [1, 4, 2], np.array([0.3, 9.1])),
        ],
    )
    @pytest.mark.parametrize("bounds", [(-1, 1), (0.1, 1.0211)])
    def test_get_random_args(self, args, num, bounds):
        """Tests the utility ``_get_random_args`` using a fixed seed."""
        seed = 921
        rnd_args = _get_random_args(args, self.interface, num, seed, bounds)
        assert len(rnd_args) == num
        np.random.seed(seed)
        for _rnd_args in rnd_args:
            expected = tuple(
                np.random.random(np.shape(arg)) * (bounds[1] - bounds[0]) + bounds[0]
                for arg in args
            )
            assert all(np.allclose(_exp, _rnd) for _exp, _rnd in zip(expected, _rnd_args))

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev, interface=interface)
    def const_circuit(x, y):
        qml.RX(0.1, wires=0)
        return qml.expval(qml.PauliZ(0))

    constant_functions = [
        const_circuit,
        lambda x: np.arange(20).reshape((2, 5, 2)),
        lambda x: (np.ones(3), -0.1),
        qml.jacobian(lambda x, y: 4 * x - 2.1 * y, argnum=[0, 1]),
    ]

    args_constant = [
        (0.1, np.array([-2.1, 0.1])),
        (1.2,),
        (np.ones((2, 3)),),
        (np.ones((3, 8)) * 0.1, -0.2 * np.ones((3, 8))),
    ]

    @qml.qnode(dev, interface=interface)
    def dependent_circuit(x, y, z):
        qml.RX(0.1, wires=0)
        qml.RY(y / 2, wires=0)
        qml.RZ(qml.math.sin(z), wires=0)
        return qml.expval(qml.PauliX(0))

    dependent_functions = [
        dependent_circuit,
        np.array,
        lambda x: np.array(x * 0.0),
        lambda x: (1 + qml.math.tanh(100 * x)) / 2,
        *dependent_lambdas,
    ]

    args_dependent = [
        (0.1, np.array(-2.1), -0.9),
        (-4.1,),
        (-4.1,),
        (np.ones((3, 8)) * 1.1,),
        *args_dependent_lambdas,
    ]

    @pytest.mark.parametrize("func, args", zip(constant_functions, args_constant))
    def test_independent(self, func, args):
        """Tests that an independent function is correctly detected as such."""
        assert is_independent(func, self.interface, args)

    @pytest.mark.parametrize("func, args", zip(dependent_functions, args_dependent))
    def test_dependent(self, func, args):
        """Tests that a dependent function is correctly detected as such."""
        assert not is_independent(func, self.interface, args)

    @pytest.mark.xfail
    @pytest.mark.parametrize("func, args", zip(overlooked_lambdas, args_overlooked_lambdas))
    def test_overlooked_dependence(self, func, args):
        """Test that particular functions that are dependent on the input
        are overlooked."""
        assert not is_independent(func, self.interface, args)

    def test_kwargs_are_considered(self):
        """Tests that kwargs are taken into account when checking
        independence of outputs."""
        f = lambda x, kw=False: 0.1 * x if kw else 0.2
        jac = qml.jacobian(f, argnum=0)
        args = (0.2,)
        assert is_independent(f, self.interface, args)
        assert not is_independent(f, self.interface, args, {"kw": True})
        assert is_independent(jac, self.interface, args, {"kw": True})

    def test_no_trainable_params_deprecation_grad(self, recwarn):
        """Tests that no deprecation warning arises when using qml.grad with
        qml.math.is_independent.
        """

        def lin(x):
            return pnp.sum(x)

        x = pnp.array([0.2, 9.1, -3.2], requires_grad=True)
        jac = qml.grad(lin)
        assert qml.math.is_independent(jac, "autograd", (x,), {})

        assert len(recwarn) == 0

    def test_no_trainable_params_deprecation_jac(self, recwarn):
        """Tests that no deprecation arises when using qml.jacobian with
        qml.math.is_independent."""

        def lin(x, weights=None):
            return np.dot(x, weights)

        x = pnp.array([0.2, 9.1, -3.2], requires_grad=True)
        weights = pnp.array([1.1, -0.7, 1.8], requires_grad=True)
        jac = qml.jacobian(lin)
        assert qml.math.is_independent(jac, "autograd", (x,), {"weights": weights})
        assert len(recwarn) == 0


class TestIsIndependentJax:
    """Tests for is_independent, which tests a function to be
    independent of its inputs, using JAX."""

    interface = "jax"

    @pytest.mark.parametrize("num", [0, 1, 2])
    @pytest.mark.parametrize(
        "args",
        [
            (0.2,),
            (1.1, 3.2, 0.2),
            (np.array([[0, 9.2], [-1.2, 3.2]]),),
            (0.3, [1, 4, 2], np.array([0.3, 9.1])),
        ],
    )
    @pytest.mark.parametrize("bounds", [(-1, 1), (0.1, 1.0211)])
    def test_get_random_args(self, args, num, bounds):
        """Tests the utility ``_get_random_args`` using a fixed seed."""
        seed = 921
        rnd_args = _get_random_args(args, self.interface, num, seed, bounds)
        assert len(rnd_args) == num
        np.random.seed(seed)
        for _rnd_args in rnd_args:
            expected = tuple(
                np.random.random(np.shape(arg)) * (bounds[1] - bounds[0]) + bounds[0]
                for arg in args
            )
            assert all(np.allclose(_exp, _rnd) for _exp, _rnd in zip(expected, _rnd_args))

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev, interface=interface)
    def const_circuit(x, y):
        qml.RX(0.1, wires=0)
        return qml.expval(qml.PauliZ(0))

    constant_functions = [
        const_circuit,
        lambda x: np.arange(20).reshape((2, 5, 2)),
        lambda x: (np.ones(3), -0.1),
        jax.jacobian(lambda x, y: 4.0 * x - 2.1 * y, argnums=[0, 1]),
    ]

    args_constant = [
        (0.1, np.array([-2.1, 0.1])),
        (1.2,),
        (np.ones((2, 3)),),
        (jax.numpy.ones((3, 8)) * 0.1, -0.2 * jax.numpy.ones((3, 8))),
    ]

    @qml.qnode(dev, interface=interface)
    def dependent_circuit(x, y, z):
        qml.RX(0.1, wires=0)
        qml.RY(y / 2, wires=0)
        qml.RZ(qml.math.sin(z), wires=0)
        return qml.expval(qml.PauliX(0))

    dependent_functions = [
        dependent_circuit,
        jax.numpy.array,
        lambda x: (1 + qml.math.tanh(1000 * x)) / 2,
        *dependent_lambdas,
    ]

    args_dependent = [
        (0.1, np.array(-2.1), -0.9),
        (-4.1,),
        (jax.numpy.ones((3, 8)) * 1.1,),
        *args_dependent_lambdas,
    ]

    @pytest.mark.parametrize("func, args", zip(constant_functions, args_constant))
    def test_independent(self, func, args):
        """Tests that an independent function is correctly detected as such."""
        assert is_independent(func, self.interface, args)

    @pytest.mark.parametrize("func, args", zip(dependent_functions, args_dependent))
    def test_dependent(self, func, args):
        """Tests that a dependent function is correctly detected as such."""
        assert not is_independent(func, self.interface, args)

    @pytest.mark.xfail
    @pytest.mark.parametrize("func, args", zip(overlooked_lambdas, args_overlooked_lambdas))
    def test_overlooked_dependence(self, func, args):
        """Test that particular functions that are dependent on the input
        are overlooked."""
        assert not is_independent(func, self.interface, args)

    def test_kwargs_are_considered(self):
        """Tests that kwargs are taken into account when checking
        independence of outputs."""
        f = lambda x, kw=False: 0.1 * x if kw else 0.2
        jac = jax.jacobian(f, argnums=0)
        args = (0.2,)
        assert is_independent(f, self.interface, args)
        assert not is_independent(f, self.interface, args, {"kw": True})
        assert is_independent(jac, self.interface, args, {"kw": True})


class TestIsIndependentTensorflow:
    """Tests for is_independent, which tests a function to be
    independent of its inputs, using Tensorflow."""

    interface = "tf"

    @pytest.mark.parametrize("num", [0, 1, 2])
    @pytest.mark.parametrize(
        "args",
        [
            (tf.Variable(0.2),),
            (tf.Variable(1.1), tf.constant(3.2), tf.Variable(0.2)),
            (tf.Variable(np.array([[0, 9.2], [-1.2, 3.2]])),),
            (tf.Variable(0.3), [1, 4, 2], tf.Variable(np.array([0.3, 9.1]))),
        ],
    )
    @pytest.mark.parametrize("bounds", [(-1, 1), (0.1, 1.0211)])
    def test_get_random_args(self, args, num, bounds):
        """Tests the utility ``_get_random_args`` using a fixed seed."""
        tf = pytest.importorskip("tensorflow")
        seed = 921
        rnd_args = _get_random_args(args, self.interface, num, seed, bounds)
        assert len(rnd_args) == num
        tf.random.set_seed(seed)
        for _rnd_args in rnd_args:
            expected = tuple(
                tf.random.uniform(tf.shape(arg)) * (bounds[1] - bounds[0]) + bounds[0]
                for arg in args
            )
            expected = tuple(
                tf.Variable(_exp) if isinstance(_arg, tf.Variable) else _exp
                for _arg, _exp in zip(args, expected)
            )
            assert all(np.allclose(_exp, _rnd) for _exp, _rnd in zip(expected, _rnd_args))

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev, interface=interface)
    def const_circuit(x, y):
        qml.RX(0.1, wires=0)
        return qml.expval(qml.PauliZ(0))

    constant_functions = [
        const_circuit,
        lambda x: np.arange(20).reshape((2, 5, 2)),
        lambda x: (np.ones(3), np.array(-0.1)),
    ]

    args_constant = [
        (0.1, np.array([-2.1, 0.1])),
        (1.2,),
        (np.ones((2, 3)),),
    ]

    @qml.qnode(dev, interface=interface)
    def dependent_circuit(x, y, z):
        qml.RX(0.1, wires=0)
        qml.RY(y / 2, wires=0)
        qml.RZ(qml.math.sin(z), wires=0)
        return qml.expval(qml.PauliX(0))

    dependent_functions = [
        dependent_circuit,
        lambda x: (1 + qml.math.tanh(1000 * x)) / 2,
        *dependent_lambdas,
    ]

    args_dependent = [
        (tf.Variable(0.1), np.array(-2.1), tf.Variable(-0.9)),
        (
            tf.Variable(
                np.ones((3, 8)) * 1.1,
            ),
        ),
        *args_dependent_lambdas,
    ]

    @pytest.mark.parametrize("func, args", zip(constant_functions, args_constant))
    def test_independent(self, func, args):
        """Tests that an independent function is correctly detected as such."""
        args = tuple([tf.Variable(_arg) for _arg in args])
        assert is_independent(func, self.interface, args)

    @pytest.mark.parametrize("func, args", zip(dependent_functions, args_dependent))
    def test_dependent(self, func, args):
        """Tests that a dependent function is correctly detected as such."""
        args = tuple([tf.Variable(_arg) for _arg in args])
        # Filter out functions with TF-incompatible output format
        out = func(*args)
        if not isinstance(out, tf.Tensor):
            try:
                _func = lambda *args: tf.Variable(func(*args))
                assert not is_independent(_func, self.interface, args)
            except:
                pytest.skip()
        else:
            assert not is_independent(func, self.interface, args)

    @pytest.mark.xfail
    @pytest.mark.parametrize("func, args", zip(overlooked_lambdas, args_overlooked_lambdas))
    def test_overlooked_dependence(self, func, args):
        """Test that particular functions that are dependent on the input
        are overlooked."""
        assert not is_independent(func, self.interface, args)

    def test_kwargs_are_considered(self):
        """Tests that kwargs are taken into account when checking
        independence of outputs."""
        f = lambda x, kw=False: 0.1 * x if kw else tf.constant(0.2)

        def _jac(x, kw):
            with tf.GradientTape() as tape:
                out = f(x, kw)
            return tape.jacobian(out, x)

        args = (tf.Variable(0.2),)
        assert is_independent(f, self.interface, args)
        assert not is_independent(f, self.interface, args, {"kw": True})
        assert is_independent(_jac, self.interface, args, {"kw": True})


class TestIsIndependentTorch:
    """Tests for is_independent, which tests a function to be
    independent of its inputs, using PyTorch."""

    interface = "torch"

    @pytest.mark.parametrize("num", [0, 1, 2])
    @pytest.mark.parametrize(
        "args",
        [
            (torch.tensor(0.2),),
            (1.1, 3.2, torch.tensor(0.2)),
            (torch.tensor([[0, 9.2], [-1.2, 3.2]]),),
            (0.3, torch.tensor([1, 4, 2]), torch.tensor([0.3, 9.1])),
        ],
    )
    @pytest.mark.parametrize("bounds", [(-1, 1), (0.1, 1.0211)])
    def test_get_random_args(self, args, num, bounds):
        """Tests the utility ``_get_random_args`` using a fixed seed."""
        torch = pytest.importorskip("torch")
        seed = 921
        rnd_args = _get_random_args(args, self.interface, num, seed, bounds)
        assert len(rnd_args) == num
        torch.random.manual_seed(seed)
        for _rnd_args in rnd_args:
            expected = tuple(
                torch.rand(np.shape(arg)) * (bounds[1] - bounds[0]) + bounds[0] for arg in args
            )
            assert all(np.allclose(_exp, _rnd) for _exp, _rnd in zip(expected, _rnd_args))

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev, interface=interface)
    def const_circuit(x, y):
        qml.RX(0.1, wires=0)
        return qml.expval(qml.PauliZ(0))

    constant_functions = [
        const_circuit,
        lambda x: np.arange(20).reshape((2, 5, 2)),
        lambda x: (np.ones(3), -0.1),
    ]

    args_constant = [
        (0.1, torch.tensor([-2.1, 0.1])),
        (1.2,),
        (torch.ones((2, 3)),),
    ]

    @qml.qnode(dev, interface=interface)
    def dependent_circuit(x, y, z):
        qml.RX(0.1, wires=0)
        qml.RY(y / 2, wires=0)
        qml.RZ(qml.math.sin(z), wires=0)
        return qml.expval(qml.PauliX(0))

    dependent_functions = [
        dependent_circuit,
        torch.as_tensor,
        lambda x: (1 + qml.math.tanh(1000 * x)) / 2,
        *dependent_lambdas,
    ]

    args_dependent = [
        (0.1, torch.tensor(-2.1), -0.9),
        (-4.1,),
        (torch.ones((3, 8)) * 1.1,),
        *args_dependent_lambdas,
    ]

    dependent_expect_torch_fail = [False, False, False, *lambdas_expect_torch_fail]

    @pytest.mark.parametrize("func, args", zip(constant_functions, args_constant))
    def test_independent(self, func, args):
        """Tests that an independent function is correctly detected as such."""
        with pytest.warns(UserWarning, match=r"The function is_independent is only available"):
            assert is_independent(func, self.interface, args)

    @pytest.mark.parametrize(
        "func, args, exp_fail",
        zip(dependent_functions, args_dependent, dependent_expect_torch_fail),
    )
    def test_dependent(self, func, args, exp_fail):
        """Tests that a dependent function is correctly detected as such."""
        with pytest.warns(UserWarning, match=r"The function is_independent is only available"):
            if exp_fail:
                assert is_independent(func, self.interface, args)
            else:
                assert not is_independent(func, self.interface, args)

    @pytest.mark.xfail
    @pytest.mark.parametrize("func, args", zip(overlooked_lambdas, args_overlooked_lambdas))
    def test_overlooked_dependence(self, func, args):
        """Test that particular functions that are dependent on the input
        are overlooked."""
        assert not is_independent(func, self.interface, args)

    def test_kwargs_are_considered(self):
        """Tests that kwargs are taken into account when checking
        independence of outputs."""
        f = lambda x, kw=False: 0.1 * x if kw else 0.2
        jac = lambda x, kw: torch.autograd.functional.jacobian(lambda x: f(x, kw), x)
        args = (torch.tensor(0.2),)
        with pytest.warns(UserWarning, match=r"The function is_independent is only available"):
            assert is_independent(f, self.interface, args)
        with pytest.warns(UserWarning, match=r"The function is_independent is only available"):
            assert not is_independent(f, self.interface, args, {"kw": True})
        with pytest.warns(UserWarning, match=r"The function is_independent is only available"):
            assert is_independent(jac, self.interface, args, {"kw": True})


class TestOther:
    """Other tests for is_independent."""

    def test_unknown_interface(self):
        """Test that an error is raised if an unknown interface is requested."""
        with pytest.raises(ValueError, match="Unknown interface: hello"):
            is_independent(lambda x: x, "hello", (0.1,))
