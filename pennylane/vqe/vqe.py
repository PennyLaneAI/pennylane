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


class ExpvalCost:
    """
    Create a cost function that gives the expectation value of an input Hamiltonian.

    This cost function is useful for a range of problems including VQE and QAOA.

    .. warning::
        ``ExpvalCost`` is deprecated. Instead, it is recommended to simply
        pass Hamiltonians to the :func:`~pennylane.expval` function inside QNodes.

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

        if not callable(ansatz):
            raise ValueError("Could not create QNodes. The template is not a callable function.")

        self.hamiltonian = hamiltonian
        """Hamiltonian: the input Hamiltonian."""

        self._multiple_devices = isinstance(device, Sequence)
        """Bool: Records if multiple devices are input"""

        self.qnodes = []
        """The QNodes to be evaluated."""

        if optimize:
            if self._multiple_devices:
                raise ValueError("Using multiple devices is not supported when optimize=True")

            hamiltonian.compute_grouping()

            @qml.qnode(device, interface=interface, diff_method=diff_method, **kwargs)
            def circuit(params, **circuit_kwargs):
                ansatz(params, wires=device.wires, **circuit_kwargs)
                return qml.expval(hamiltonian)

            self.qnodes.append(circuit)
            self.cost_fn = self.qnodes[0]

        else:
            coeffs, observables = hamiltonian.terms()
            if not self._multiple_devices:
                device = [device] * len(coeffs)

            for obs, dev in zip(observables, device):
                wires = dev.wires

                @qml.qnode(
                    dev,  # pylint: disable=cell-var-from-loop
                    interface=interface,
                    diff_method=diff_method,
                    **kwargs,
                )
                def circuit(params, _wires=wires, _obs=obs, **circuit_kwargs):
                    ansatz(params, wires=_wires, **circuit_kwargs)
                    return qml.expval(_obs)

                self.qnodes.append(circuit)

            def cost_fn(*args, **kwargs):
                res = [q(*args, **kwargs) for q in self.qnodes]
                # pylint: disable=no-member
                res = [
                    qml.math.stack(r)
                    if isinstance(r, (tuple, qml.numpy.builtins.SequenceBox))
                    else r
                    for r in res
                ]
                return sum(c * q for c, q in zip(coeffs, res))

            self.cost_fn = cost_fn

    def __call__(self, *args, **kwargs):
        return self.cost_fn(*args, **kwargs)
