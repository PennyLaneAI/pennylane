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
This submodule contains functionality for running Variational Quantum Eigensolver (VQE)
computations using PennyLane.
"""
import warnings

# pylint: disable=too-many-arguments, too-few-public-methods
from collections.abc import Sequence

import pennylane as qml
from pennylane import numpy as np


class ExpvalCost:
    """
    Create a cost function that gives the expectation value of an input Hamiltonian.

    This cost function is useful for a range of problems including VQE and QAOA.

    .. warning::
        ``ExpvalCost`` is deprecated. Instead, it is recommended to simply
        pass Hamiltonians to the :func:`~.expval` function inside QNodes.

        .. code-block:: python

            @qml.qnode(dev)
            def ansatz(params):
                some_qfunc(params)
                return qml.expval(Hamiltonian)

        In order to optimize the Hamiltonian evaluation taking into account commuting terms, use the ``grouping_type`` keyword in :class:`~.Hamiltonian`.

    Args:
        ansatz (callable): The ansatz for the circuit before the final measurement step.
            Note that the ansatz **must** have the following signature:

            .. code-block:: python

                ansatz(params, wires, **kwargs)

            where ``params`` are the trainable weights of the variational circuit,
            ``wires`` is the wires the circuit acts on, and ``kwargs`` are any additional
            keyword arguments that need to be passed to the template.
        hamiltonian (~.Hamiltonian): Hamiltonian operator whose expectation value should be measured
        device (Device, Sequence[Device]): Corresponding device(s) where the resulting
            cost function should be executed. This can either be a single device, or a list
            of devices of length matching the number of terms in the Hamiltonian.
        interface (str, None): Which interface to use.
            This affects the types of objects that can be passed to/returned to the cost function.
            Supports all interfaces supported by the :func:`~.qnode` decorator.
        diff_method (str, None): The method of differentiation to use with the created cost function.
            Supports all differentiation methods supported by the :func:`~.qnode` decorator.
        optimize (bool): Whether to optimize the observables composing the Hamiltonian by
            separating them into qubit-wise commuting groups. Each group can then be executed
            within a single QNode, resulting in fewer QNodes to evaluate.

    Returns:
        callable: a cost function with signature ``cost_fn(params, **kwargs)`` that evaluates
        the expectation of the Hamiltonian on the provided device(s)

    .. seealso:: :class:`~.Hamiltonian`, :func:`~.molecular_hamiltonian`, :func:`~.map`, :func:`~.dot`

    **Example:**

    To construct an ``ExpvalCost`` cost function, we require a Hamiltonian to measure, and an ansatz
    for our variational circuit.

    We can construct a Hamiltonian manually,

    .. code-block:: python

        coeffs = [0.2, -0.543]
        obs = [
            qml.PauliX(0) @ qml.PauliZ(1) @ qml.PauliY(3),
            qml.PauliZ(0) @ qml.Hadamard(2)
        ]
        H = qml.Hamiltonian(coeffs, obs)

    Alternatively, the :func:`~.molecular_hamiltonian` function from the
    :doc:`/introduction/chemistry` module can be used to generate a molecular Hamiltonian.

    Once we have our Hamiltonian, we can select an ansatz and construct
    the cost function.

    >>> ansatz = qml.templates.StronglyEntanglingLayers
    >>> dev = qml.device("default.qubit", wires=4)
    >>> cost = qml.ExpvalCost(ansatz, H, dev, interface="torch")
    >>> params = torch.rand([2, 4, 3])
    >>> cost(params)
    tensor(-0.2316, dtype=torch.float64)

    The cost function can then be minimized using any gradient descent-based
    :doc:`optimizer </introduction/interfaces>`.

    .. details::
        :title: Usage Details

        **Optimizing observables:**

        Setting ``optimize=True`` can be used to decrease the number of device executions. The
        observables composing the Hamiltonian can be separated into groups that are qubit-wise
        commuting using the :mod:`~.grouping` module. These groups can be executed together on a
        *single* qnode, resulting in a lower device overhead:

        .. code-block:: python

            commuting_obs = [qml.PauliX(0), qml.PauliX(0) @ qml.PauliZ(1)]
            H = qml.Hamiltonian([1, 1], commuting_obs)

            dev = qml.device("default.qubit", wires=2)
            ansatz = qml.templates.StronglyEntanglingLayers

            cost_opt = qml.ExpvalCost(ansatz, H, dev, optimize=True)
            cost_no_opt = qml.ExpvalCost(ansatz, H, dev, optimize=False)

            shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=3, n_wires=2)
            params = np.random.random(shape)

        Grouping these commuting observables leads to fewer device executions:

        >>> cost_opt(params)
        >>> ex_opt = dev.num_executions
        >>> cost_no_opt(params)
        >>> ex_no_opt = dev.num_executions - ex_opt
        >>> print("Number of executions:", ex_no_opt)
        Number of executions: 2
        >>> print("Number of executions (optimized):", ex_opt)
        Number of executions (optimized): 1
    """

    def __init__(
        self,
        ansatz,
        hamiltonian,
        device,
        interface="autograd",
        diff_method="best",
        optimize=False,
        **kwargs,
    ):
        warnings.warn(
            "ExpvalCost is deprecated, use qml.expval() instead. "
            "For optimizing Hamiltonian measurements with measuring commuting "
            "terms in parallel, use the grouping_type keyword in qml.Hamiltonian.",
            UserWarning,
        )

        if kwargs.get("measure", "expval") != "expval":
            raise ValueError("ExpvalCost can only be used to construct sums of expectation values.")

        coeffs, observables = hamiltonian.terms()

        self.hamiltonian = hamiltonian
        """Hamiltonian: the input Hamiltonian."""

        self.qnodes = None
        """QNodeCollection: The QNodes to be evaluated. Each QNode corresponds to the expectation
        value of each observable term after applying the circuit ansatz."""

        self._multiple_devices = isinstance(device, Sequence)
        """Bool: Records if multiple devices are input"""

        if np.isclose(qml.math.toarray(qml.math.count_nonzero(coeffs)), 0):
            self.cost_fn = lambda *args, **kwargs: np.array(0)
            return

        self._optimize = optimize

        self.qnodes = qml.map(
            ansatz, observables, device, interface=interface, diff_method=diff_method, **kwargs
        )

        if self._optimize:

            if self._multiple_devices:
                raise ValueError("Using multiple devices is not supported when optimize=True")

            obs_groupings, coeffs_groupings = qml.grouping.group_observables(observables, coeffs)
            d = device[0] if self._multiple_devices else device
            w = d.wires.tolist()

            @qml.qnode(device, interface=interface, diff_method=diff_method, **kwargs)
            def circuit(*qnode_args, obs, **qnode_kwargs):
                """Converting ansatz into a full circuit including measurements"""
                ansatz(*qnode_args, wires=w, **qnode_kwargs)
                return [qml.expval(o) for o in obs]

            def cost_fn(*qnode_args, **qnode_kwargs):
                """Combine results from grouped QNode executions with grouped coefficients"""
                if device.shot_vector:
                    shots_batch = sum(i[1] for i in device.shot_vector)

                    total = [0] * shots_batch

                    for o, c in zip(obs_groupings, coeffs_groupings):
                        res = circuit(*qnode_args, obs=o, **qnode_kwargs)

                        for i, batch_res in enumerate(res):
                            total[i] += sum(batch_res * c)
                else:
                    total = 0
                    for o, c in zip(obs_groupings, coeffs_groupings):
                        res = circuit(*qnode_args, obs=o, **qnode_kwargs)
                        total += sum(r * c_ for r, c_ in zip(res, c))
                return total

            self.cost_fn = cost_fn

        else:
            self.cost_fn = qml.dot(coeffs, self.qnodes)

    def __call__(self, *args, **kwargs):
        return self.cost_fn(*args, **kwargs)


class VQECost(ExpvalCost):
    """Create a cost function that gives the expectation value of an input Hamiltonian.

    .. warning::
        Use of :class:`~.VQECost` is deprecated and should be replaced with
        :class:`~.ExpvalCost`.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "Use of VQECost is deprecated and should be replaced with ExpvalCost", UserWarning, 2,
        )
        super().__init__(*args, **kwargs)
