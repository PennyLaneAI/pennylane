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
"""QNode transforms for the quantum information quantities."""
# pylint: disable=import-outside-toplevel, not-callable
import warnings
from collections.abc import Callable, Sequence
from functools import partial

import pennylane as qml
from pennylane import transform
from pennylane.devices import DefaultMixed
from pennylane.measurements import DensityMatrixMP, StateMP
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.typing import PostprocessingFn


@partial(transform, final_transform=True)
def reduced_dm(tape: QuantumScript, wires, **kwargs) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Compute the reduced density matrix from a :class:`~.QNode` returning
    :func:`~pennylane.state`.

    .. warning::

        The ``qml.qinfo.reduced_dm`` transform is deprecated and will be removed in v0.40. Instead include
        the :func:`pennylane.density_matrix` measurement process in the return line of your QNode.

    Args:
        tape (QuantumTape or QNode or Callable)): A quantum circuit returning :func:`~pennylane.state`.
        wires (Sequence(int)): List of wires in the considered subsystem.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will provide the reduced density matrix in the form of a tensor.

    **Example**

    .. code-block:: python

        import numpy as np

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.IsingXX(x, wires=[0,1])
            return qml.state()

    >>> transformed_circuit = reduced_dm(circuit, wires=[0])
    >>> transformed_circuit(np.pi/2)
    tensor([[0.5+0.j, 0. +0.j],
            [0. +0.j, 0.5+0.j]], requires_grad=True)

    This is equivalent to the state of the wire ``0`` after measuring the wire ``1``:

    .. code-block:: python

        @qml.qnode(dev)
        def measured_circuit(x):
            qml.IsingXX(x, wires=[0,1])
            m = qml.measure(1)
            return qml.density_matrix(wires=[0]), qml.probs(op=m)

    >>> dm, probs = measured_circuit(np.pi/2)
    >>> dm
    tensor([[0.5+0.j, 0. +0.j],
            [0. +0.j, 0.5+0.j]], requires_grad=True)
    >>> probs
    tensor([0.5, 0.5], requires_grad=True)

    .. seealso:: :func:`pennylane.density_matrix` and :func:`pennylane.math.reduce_dm`
    """

    warnings.warn(
        "The qml.qinfo.reduced_dm transform is deprecated and will be removed "
        "in v0.40. Instead include the qml.density_matrix measurement process in the "
        "return line of your QNode.",
        qml.PennyLaneDeprecationWarning,
    )

    # device_wires is provided by the custom QNode transform
    all_wires = kwargs.get("device_wires", tape.wires)
    wire_map = {w: i for i, w in enumerate(all_wires)}
    indices = [wire_map[w] for w in wires]

    measurements = tape.measurements
    if len(measurements) != 1 or not isinstance(measurements[0], StateMP):
        raise ValueError("The qfunc measurement needs to be State.")

    def processing_fn(res):
        # device is provided by the custom QNode transform
        device = kwargs.get("device", None)
        c_dtype = getattr(device, "C_DTYPE", "complex128")

        # determine the density matrix
        dm_func = (
            qml.math.reduce_dm
            if isinstance(measurements[0], DensityMatrixMP) or isinstance(device, DefaultMixed)
            else qml.math.reduce_statevector
        )
        density_matrix = dm_func(res[0], indices=indices, c_dtype=c_dtype)

        return density_matrix

    return [tape], processing_fn


@reduced_dm.custom_qnode_transform
def _reduced_dm_qnode(self, qnode, targs, tkwargs):
    if tkwargs.get("device", False):
        raise ValueError(
            "Cannot provide a 'device' value directly to the reduced_dm decorator when "
            "transforming a QNode."
        )
    if tkwargs.get("device_wires", None):
        raise ValueError(
            "Cannot provide a 'device_wires' value directly to the reduced_dm decorator when "
            "transforming a QNode."
        )

    tkwargs.setdefault("device", qnode.device)
    tkwargs.setdefault("device_wires", qnode.device.wires)
    return self.default_qnode_transform(qnode, targs, tkwargs)


@partial(transform, final_transform=True)
def purity(tape: QuantumScript, wires, **kwargs) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    r"""Compute the purity of a :class:`~.QuantumTape` returning :func:`~pennylane.state`.

    .. math::
        \gamma = \text{Tr}(\rho^2)

    where :math:`\rho` is the density matrix. The purity of a normalized quantum state satisfies
    :math:`\frac{1}{d} \leq \gamma \leq 1`, where :math:`d` is the dimension of the Hilbert space.
    A pure state has a purity of 1.

    It is possible to compute the purity of a sub-system from a given state. To find the purity of
    the overall state, include all wires in the ``wires`` argument.

    .. warning::

        The ``qml.qinfo.purity transform`` is deprecated and will be removed in v0.40. Instead include
        the :func:`pennylane.purity` measurement process in the return line of your QNode.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit object returning a :func:`~pennylane.state`.
        wires (Sequence(int)): List of wires in the considered subsystem.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[.QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will provide the purity in the form of a tensor.

    **Example**

    .. code-block:: python

        dev = qml.device("default.mixed", wires=2)

        @qml.qnode(dev)
        def noisy_circuit(p):
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.BitFlip(p, wires=0)
            qml.BitFlip(p, wires=1)
            return qml.state()

        @qml.qnode(dev)
        def circuit(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

    >>> purity(noisy_circuit, wires=[0, 1])(0.2)
    0.5648000000000398
    >>> purity(circuit, wires=[0])(np.pi / 2)
    0.5
    >>> purity(circuit, wires=[0, 1])(np.pi / 2)
    1.0

    .. seealso:: :func:`pennylane.math.purity`
    """

    warnings.warn(
        "The qml.qinfo.purity transform is deprecated and will be removed "
        "in v0.40. Instead include the qml.purity measurement process in the "
        "return line of your QNode.",
        qml.PennyLaneDeprecationWarning,
    )

    # device_wires is provided by the custom QNode transform
    all_wires = kwargs.get("device_wires", tape.wires)
    wire_map = {w: i for i, w in enumerate(all_wires)}
    indices = [wire_map[w] for w in wires]

    # Check measurement
    measurements = tape.measurements
    if len(measurements) != 1 or not isinstance(measurements[0], StateMP):
        raise ValueError("The qfunc return type needs to be a state.")

    def processing_fn(res):
        # device is provided by the custom QNode transform
        device = kwargs.get("device", None)
        c_dtype = getattr(device, "C_DTYPE", "complex128")

        # determine the density matrix
        density_matrix = (
            res[0]
            if isinstance(measurements[0], DensityMatrixMP) or isinstance(device, DefaultMixed)
            else qml.math.dm_from_state_vector(res[0], c_dtype=c_dtype)
        )

        return qml.math.purity(density_matrix, indices, c_dtype=c_dtype)

    return [tape], processing_fn


@purity.custom_qnode_transform
def _purity_qnode(self, qnode, targs, tkwargs):
    if tkwargs.get("device", False):
        raise ValueError(
            "Cannot provide a 'device' value directly to the purity decorator when "
            "transforming a QNode."
        )
    if tkwargs.get("device_wires", None):
        raise ValueError(
            "Cannot provide a 'device_wires' value directly to the purity decorator when "
            "transforming a QNode."
        )

    tkwargs.setdefault("device", qnode.device)
    tkwargs.setdefault("device_wires", qnode.device.wires)
    return self.default_qnode_transform(qnode, targs, tkwargs)


@partial(transform, final_transform=True)
def vn_entropy(
    tape: QuantumScript, wires: Sequence[int], base: float = None, **kwargs
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    r"""Compute the Von Neumann entropy from a :class:`.QuantumTape` returning a :func:`~pennylane.state`.

    .. math::
        S( \rho ) = -\text{Tr}( \rho \log ( \rho ))

    .. warning::

        The ``qml.qinfo.vn_entropy`` transform is deprecated and will be removed in v0.40. Instead include
        the :func:`pennylane.vn_entropy` measurement process in the return line of your QNode.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit returning a :func:`~pennylane.state`.
        wires (Sequence(int)): List of wires in the considered subsystem.
        base (float): Base for the logarithm, default is None the natural logarithm is used in this case.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will provide the Von Neumann entropy in the form of a tensor.

    **Example**

    It is possible to obtain the entropy of a subsystem from a :class:`.QNode` returning a :func:`~pennylane.state`.

    .. code-block:: python

        dev = qml.device("default.qubit", wires=2)
        @qml.qnode(dev)
        def circuit(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

    >>> vn_entropy(circuit, wires=[0])(np.pi/2)
    0.6931471805599453

    The function is differentiable with backpropagation for all interfaces, e.g.:

    >>> param = np.array(np.pi/4, requires_grad=True)
    >>> qml.grad(vn_entropy(circuit, wires=[0]))(param)
    tensor(0.62322524, requires_grad=True)

    .. seealso:: :func:`pennylane.math.vn_entropy` and :func:`pennylane.vn_entropy`
    """

    warnings.warn(
        "The qml.qinfo.vn_entropy transform is deprecated and will be removed "
        "in v0.40. Instead include the qml.vn_entropy measurement process in the "
        "return line of your QNode.",
        qml.PennyLaneDeprecationWarning,
    )

    # device_wires is provided by the custom QNode transform
    all_wires = kwargs.get("device_wires", tape.wires)
    wire_map = {w: i for i, w in enumerate(all_wires)}
    indices = [wire_map[w] for w in wires]

    measurements = tape.measurements
    if len(measurements) != 1 or not isinstance(measurements[0], StateMP):
        raise ValueError("The qfunc return type needs to be a state.")

    def processing_fn(res):
        # device is provided by the custom QNode transform
        device = kwargs.get("device", None)
        c_dtype = getattr(device, "C_DTYPE", "complex128")

        # determine if the measurement is a state vector or a density matrix
        if not isinstance(measurements[0], DensityMatrixMP) and not isinstance(
            device, DefaultMixed
        ):  # Compute entropy from state vector
            if len(wires) == len(all_wires):
                # The subsystem has all wires, so the entropy is 0
                return 0.0

            density_matrix = qml.math.dm_from_state_vector(res[0], c_dtype=c_dtype)
            entropy = qml.math.vn_entropy(density_matrix, indices, base, c_dtype=c_dtype)
            return entropy

        # Compute entropy from density matrix
        entropy = qml.math.vn_entropy(res[0], indices, base, c_dtype)
        return entropy

    return [tape], processing_fn


@vn_entropy.custom_qnode_transform
def _vn_entropy_qnode(self, qnode, targs, tkwargs):
    if tkwargs.get("device", False):
        raise ValueError(
            "Cannot provide a 'device' value directly to the vn_entropy decorator when "
            "transforming a QNode."
        )
    if tkwargs.get("device_wires", None):
        raise ValueError(
            "Cannot provide a 'device_wires' value directly to the vn_entropy decorator when "
            "transforming a QNode."
        )

    tkwargs.setdefault("device", qnode.device)
    tkwargs.setdefault("device_wires", qnode.device.wires)
    return self.default_qnode_transform(qnode, targs, tkwargs)


def _bipartite_qinfo_transform(
    transform_func: Callable,
    tape: QuantumScript,
    wires0: Sequence[int],
    wires1: Sequence[int],
    base: float = None,
    **kwargs,
):

    # device_wires is provided by the custom QNode transform
    all_wires = kwargs.get("device_wires", tape.wires)
    wire_map = {w: i for i, w in enumerate(all_wires)}
    indices0 = [wire_map[w] for w in wires0]
    indices1 = [wire_map[w] for w in wires1]

    # Check measurement
    measurements = tape.measurements
    if len(measurements) != 1 or not isinstance(measurements[0], StateMP):
        raise ValueError("The qfunc return type needs to be a state.")

    def processing_fn(res):
        # device is provided by the custom QNode transform
        device = kwargs.get("device", None)
        c_dtype = getattr(device, "C_DTYPE", "complex128")

        density_matrix = (
            res[0]
            if isinstance(measurements[0], DensityMatrixMP) or isinstance(device, DefaultMixed)
            else qml.math.dm_from_state_vector(res[0], c_dtype=c_dtype)
        )
        entropy = transform_func(density_matrix, indices0, indices1, base=base, c_dtype=c_dtype)
        return entropy

    return [tape], processing_fn


@partial(transform, final_transform=True)
def mutual_info(
    tape: QuantumScript, wires0: Sequence[int], wires1: Sequence[int], base: float = None, **kwargs
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    r"""Compute the mutual information from a :class:`.QuantumTape` returning a :func:`~pennylane.state`:

    .. math::

        I(A, B) = S(\rho^A) + S(\rho^B) - S(\rho^{AB})

    where :math:`S` is the von Neumann entropy.

    The mutual information is a measure of correlation between two subsystems.
    More specifically, it quantifies the amount of information obtained about
    one system by measuring the other system.

    .. warning::

        The ``qml.qinfo.mutual_info`` transform is deprecated and will be removed in v0.40. Instead include
        the :func:`pennylane.mutual_info` measurement process in the return line of your QNode.

    Args:
        qnode (QNode or QuantumTape or Callable): A quantum circuit returning a :func:`~pennylane.state`.
        wires0 (Sequence(int)): List of wires in the first subsystem.
        wires1 (Sequence(int)): List of wires in the second subsystem.
        base (float): Base for the logarithm. If None, the natural logarithm is used.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will provide the mutual information in the form of a tensor.

    **Example**

    It is possible to obtain the mutual information of two subsystems from a
    :class:`.QNode` returning a :func:`~pennylane.state`.

    .. code-block:: python

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

    >>> mutual_info_circuit = qinfo.mutual_info(circuit, wires0=[0], wires1=[1])
    >>> mutual_info_circuit(np.pi/2)
    1.3862943611198906
    >>> x = np.array(0.4, requires_grad=True)
    >>> mutual_info_circuit(x)
    0.3325090393262875
    >>> qml.grad(mutual_info_circuit)(np.array(0.4, requires_grad=True))
    tensor(1.24300677, requires_grad=True)

    .. seealso:: :func:`~.qinfo.vn_entropy`, :func:`pennylane.math.mutual_info` and :func:`pennylane.mutual_info`
    """

    warnings.warn(
        "The qml.qinfo.mutual_info transform is deprecated and will be removed "
        "in v0.40. Instead include the qml.mutual_info measurement process in the "
        "return line of your QNode.",
        qml.PennyLaneDeprecationWarning,
    )

    return _bipartite_qinfo_transform(qml.math.mutual_info, tape, wires0, wires1, base, **kwargs)


@mutual_info.custom_qnode_transform
def _mutual_info_qnode(self, qnode, targs, tkwargs):
    if tkwargs.get("device", False):
        raise ValueError(
            "Cannot provide a 'device' value directly to the mutual_info decorator when "
            "transforming a QNode."
        )
    if tkwargs.get("device_wires", None):
        raise ValueError(
            "Cannot provide a 'device_wires' value directly to the mutual_info decorator when "
            "transforming a QNode."
        )

    tkwargs.setdefault("device", qnode.device)
    tkwargs.setdefault("device_wires", qnode.device.wires)
    return self.default_qnode_transform(qnode, targs, tkwargs)


@partial(transform, final_transform=True)
def vn_entanglement_entropy(
    tape, wires0: Sequence[int], wires1: Sequence[int], base: float = None, **kwargs
):
    r"""Compute the Von Neumann entanglement entropy from a circuit returning a :func:`~pennylane.state`:

    .. math::

        S(\rho_A) = -\text{Tr}[\rho_A \log \rho_A] = -\text{Tr}[\rho_B \log \rho_B] = S(\rho_B)

    where :math:`S` is the von Neumann entropy; :math:`\rho_A = \text{Tr}_B [\rho_{AB}]` and
    :math:`\rho_B = \text{Tr}_A [\rho_{AB}]` are the reduced density matrices for each partition.

    .. warning::

        The ``qml.qinfo.vn_entanglement_entropy`` transform is deprecated and will be removed in v0.40.
        See the :func:`pennylane.vn_entropy` measurement instead.

    The Von Neumann entanglement entropy is a measure of the degree of quantum entanglement between
    two subsystems constituting a pure bipartite quantum state. The entropy of entanglement is the
    Von Neumann entropy of the reduced density matrix for any of the subsystems. If it is non-zero,
    it indicates the two subsystems are entangled.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit returning a :func:`~pennylane.state`.
        wires0 (Sequence(int)): List of wires in the first subsystem.
        wires1 (Sequence(int)): List of wires in the second subsystem.
        base (float): Base for the logarithm. If None, the natural logarithm is used.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will provide the entanglement entropy in the form of a tensor.

    """

    warnings.warn(
        "The qml.qinfo.vn_entanglement_entropy transform is deprecated and will "
        "be removed in v0.40. Instead include the qml.vn_entropy measurement process "
        "on one of the subsystems.",
        qml.PennyLaneDeprecationWarning,
    )

    return _bipartite_qinfo_transform(
        qml.math.vn_entanglement_entropy, tape, wires0, wires1, base, **kwargs
    )


def fidelity(qnode0, qnode1, wires0, wires1):
    r"""Compute the fidelity for two :class:`.QNode` returning a :func:`~pennylane.state` (a state can be a state vector
    or a density matrix, depending on the device) acting on quantum systems with the same size.

    The fidelity for two mixed states given by density matrices :math:`\rho` and :math:`\sigma`
    is defined as

    .. math::
        F( \rho , \sigma ) = \text{Tr}( \sqrt{\sqrt{\rho} \sigma \sqrt{\rho}})^2

    If one of the states is pure, say :math:`\rho=\ket{\psi}\bra{\psi}`, then the expression
    for fidelity simplifies to

    .. math::
        F( \ket{\psi} , \sigma ) = \bra{\psi} \sigma \ket{\psi}

    Finally, if both states are pure, :math:`\sigma=\ket{\phi}\bra{\phi}`, then the
    fidelity is simply

    .. math::
        F( \ket{\psi} , \ket{\phi}) = \left|\braket{\psi| \phi}\right|^2

    .. note::
        The second state is coerced to the type and dtype of the first state. The fidelity is returned in the type
        of the interface of the first state.

    .. warning::

        ``qml.qinfo.fidelity`` is deprecated and will be removed in v0.40. Instead, use
        :func:`pennylane.math.fidelity`.

    Args:
        state0 (QNode): A :class:`.QNode` returning a :func:`~pennylane.state`.
        state1 (QNode): A :class:`.QNode` returning a :func:`~pennylane.state`.
        wires0 (Sequence[int]): the wires of the first subsystem
        wires1 (Sequence[int]): the wires of the second subsystem

    Returns:
        func: A function that returns the fidelity between the states outputted by the QNodes.

    **Example**

    First, let's consider two QNodes with potentially different signatures: a circuit with two parameters
    and another circuit with a single parameter. The output of the :func:`~.qinfo.fidelity` transform then requires
    two tuples to be passed as arguments, each containing the args and kwargs of their respective circuit, e.g.
    ``all_args0 = (0.1, 0.3)`` and ``all_args1 = (0.2)`` in the following case:

    .. code-block:: python

        dev = qml.device('default.qubit', wires=1)

        @qml.qnode(dev)
        def circuit_rx(x, y):
            qml.RX(x, wires=0)
            qml.RZ(y, wires=0)
            return qml.state()

        @qml.qnode(dev)
        def circuit_ry(y):
            qml.RY(y, wires=0)
            return qml.state()

    >>> qml.qinfo.fidelity(circuit_rx, circuit_ry, wires0=[0], wires1=[0])((0.1, 0.3), (0.2))
    0.9905158135644924

    It is also possible to use QNodes that do not depend on any parameters. When it is the case for the first QNode, it
    is required to pass an empty tuple as an argument for the first QNode.

    .. code-block:: python

        dev = qml.device('default.qubit', wires=1)

        @qml.qnode(dev)
        def circuit_rx():
            return qml.state()

        @qml.qnode(dev)
        def circuit_ry(x):
            qml.RY(x, wires=0)
            return qml.state()

    >>> qml.qinfo.fidelity(circuit_rx, circuit_ry, wires0=[0], wires1=[0])(None, (0.2))
    0.9900332889206207

    On the other hand, if the second QNode is the one that does not depend on parameters then a single tuple can also be
    passed:

    >>> qml.qinfo.fidelity(circuit_ry, circuit_rx, wires0=[0], wires1=[0])((0.2))
    0.9900332889206207

    The :func:`~.qinfo.fidelity` transform is also differentiable and the gradient can be obtained in the different frameworks
    with backpropagation, the following example uses ``jax`` and ``backprop``.

    .. code-block:: python

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, interface="jax")
        def circuit0(x):
            qml.RX(x, wires=0)
            return qml.state()

        @qml.qnode(dev, interface="jax")
        def circuit1():
            qml.Z(0)
            return qml.state()

    >>> jax.grad(qml.qinfo.fidelity(circuit0, circuit1, wires0=[0], wires1=[0]))((jax.numpy.array(0.3)))
    Array(-0.14776011, dtype=float64, weak_type=True)

    There is also the possibility to pass a single dictionary at the end of the tuple for fixing args,
    you can follow this example:

    .. code-block:: python

        dev = qml.device('default.qubit', wires=1)

        @qml.qnode(dev)
        def circuit_rx(x, y):
            qml.RX(x, wires=0)
            qml.RZ(y, wires=0)
            return qml.state()

        @qml.qnode(dev)
        def circuit_ry(y, use_ry):
            if use_ry:
                qml.RY(y, wires=0)
            return qml.state()

    >>> fidelity(circuit_rx, circuit_ry, wires0=[0], wires1=[0])((0.1, 0.3), (0.9, {'use_ry': True}))
    0.8208074192135424

    .. seealso:: :func:`pennylane.math.fidelity`
    """

    warnings.warn(
        "qml.qinfo.fidelity is deprecated and will be removed in v0.40. Instead, use "
        "qml.math.fidelity.",
        qml.PennyLaneDeprecationWarning,
    )

    if len(wires0) != len(wires1):
        raise qml.QuantumFunctionError("The two states must have the same number of wires.")

    # with warnings.catch_warnings():
    warnings.filterwarnings(
        action="ignore",
        message="The qml.qinfo.reduced_dm transform",
        category=qml.PennyLaneDeprecationWarning,
    )
    state_qnode0 = qml.qinfo.reduced_dm(qnode0, wires=wires0)
    state_qnode1 = qml.qinfo.reduced_dm(qnode1, wires=wires1)

    def evaluate_fidelity(all_args0=None, all_args1=None):
        """Wrapper used for evaluation of the fidelity between two states computed from QNodes. It allows giving
        the args and kwargs to each :class:`.QNode`.

        Args:
            all_args0 (tuple): Tuple containing the arguments (*args, kwargs) of the first :class:`.QNode`.
            all_args1 (tuple): Tuple containing the arguments (*args, kwargs) of the second :class:`.QNode`.

        Returns:
            float: Fidelity between two quantum states
        """
        if not isinstance(all_args0, tuple) and all_args0 is not None:
            all_args0 = (all_args0,)

        if not isinstance(all_args1, tuple) and all_args1 is not None:
            all_args1 = (all_args1,)

        # If no all_args is given, evaluate the QNode without args
        if all_args0 is not None:
            # Handle a dictionary as last argument
            if isinstance(all_args0[-1], dict):
                args0 = all_args0[:-1]
                kwargs0 = all_args0[-1]
            else:
                args0 = all_args0
                kwargs0 = {}
            state0 = state_qnode0(*args0, **kwargs0)
        else:
            # No args
            state0 = state_qnode0()

        # If no all_args is given, evaluate the QNode without args
        if all_args1 is not None:
            # Handle a dictionary as last argument
            if isinstance(all_args1[-1], dict):
                args1 = all_args1[:-1]
                kwargs1 = all_args1[-1]
            else:
                args1 = all_args1
                kwargs1 = {}
            state1 = state_qnode1(*args1, **kwargs1)
        else:
            # No args
            state1 = state_qnode1()

        # From the two generated states, compute the fidelity.
        fid = qml.math.fidelity(state0, state1)
        return fid

    return evaluate_fidelity


def relative_entropy(qnode0, qnode1, wires0, wires1):
    r"""
    Compute the relative entropy for two :class:`.QNode` returning a :func:`~pennylane.state` (a state can be a state vector
    or a density matrix, depending on the device) acting on quantum systems with the same size.

    .. math::
        S(\rho\,\|\,\sigma)=-\text{Tr}(\rho\log\sigma)-S(\rho)=\text{Tr}(\rho\log\rho)-\text{Tr}(\rho\log\sigma)
        =\text{Tr}(\rho(\log\rho-\log\sigma))

    Roughly speaking, quantum relative entropy is a measure of distinguishability between two
    quantum states. It is the quantum mechanical analog of relative entropy.

    .. warning::

        ``qml.qinfo.relative_entropy`` is deprecated and will be removed in v0.40. Instead, use
        :func:`~pennylane.math.relative_entropy`.

    Args:
        qnode0 (QNode): A :class:`.QNode` returning a :func:`~pennylane.state`.
        qnode1 (QNode): A :class:`.QNode` returning a :func:`~pennylane.state`.
        wires0 (Sequence[int]): the subsystem of the first QNode
        wires1 (Sequence[int]): the subsystem of the second QNode

    Returns:
        func: A function that takes as input the joint arguments of the two QNodes,
        and returns the relative entropy from their output states.

    **Example**

    Consider the following QNode:

    .. code-block:: python

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit(param):
            qml.RY(param, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

    The ``qml.qinfo.relative_entropy`` transform can be used to compute the relative
    entropy between the output states of the QNode:

    >>> relative_entropy_circuit = qml.qinfo.relative_entropy(circuit, circuit, wires0=[0], wires1=[0])

    The returned function takes two tuples as input, the first being the arguments to the
    first QNode and the second being the arguments to the second QNode:

    >>> x, y = np.array(0.4), np.array(0.6)
    >>> relative_entropy_circuit((x,), (y,))
    tensor(0.01775001, requires_grad=True)

    This transform is fully differentiable:

    .. code-block:: python

        def wrapper(x, y):
            return relative_entropy_circuit((x,), (y,))

    >>> wrapper(x, y)
    tensor(0.01775001, requires_grad=True)
    >>> qml.grad(wrapper)(x, y)
    (tensor(-0.16458856, requires_grad=True),
     tensor(0.16953273, requires_grad=True))
    """

    warnings.warn(
        "qml.qinfo.relative_entropy is deprecated and will be removed in v0.40. Instead, use "
        "qml.math.relative_entropy.",
        qml.PennyLaneDeprecationWarning,
    )

    if len(wires0) != len(wires1):
        raise qml.QuantumFunctionError("The two states must have the same number of wires.")

    # with warnings.catch_warnings():
    warnings.filterwarnings(
        action="ignore",
        message="The qml.qinfo.reduced_dm transform",
        category=qml.PennyLaneDeprecationWarning,
    )
    state_qnode0 = qml.qinfo.reduced_dm(qnode0, wires=wires0)
    state_qnode1 = qml.qinfo.reduced_dm(qnode1, wires=wires1)

    def evaluate_relative_entropy(all_args0=None, all_args1=None):
        """Wrapper used for evaluation of the relative entropy between two states computed from
        QNodes. It allows giving the args and kwargs to each :class:`.QNode`.

        Args:
            all_args0 (tuple): Tuple containing the arguments (*args, kwargs) of the first :class:`.QNode`.
            all_args1 (tuple): Tuple containing the arguments (*args, kwargs) of the second :class:`.QNode`.

        Returns:
            float: Relative entropy between two quantum states
        """
        if not isinstance(all_args0, tuple) and all_args0 is not None:
            all_args0 = (all_args0,)

        if not isinstance(all_args1, tuple) and all_args1 is not None:
            all_args1 = (all_args1,)

        # If no all_args is given, evaluate the QNode without args
        if all_args0 is not None:
            # Handle a dictionary as last argument
            if isinstance(all_args0[-1], dict):
                args0 = all_args0[:-1]
                kwargs0 = all_args0[-1]
            else:
                args0 = all_args0
                kwargs0 = {}
            state0 = state_qnode0(*args0, **kwargs0)
        else:
            # No args
            state0 = state_qnode0()

        # If no all_args is given, evaluate the QNode without args
        if all_args1 is not None:
            # Handle a dictionary as last argument
            if isinstance(all_args1[-1], dict):
                args1 = all_args1[:-1]
                kwargs1 = all_args1[-1]
            else:
                args1 = all_args1
                kwargs1 = {}
            state1 = state_qnode1(*args1, **kwargs1)
        else:
            # No args
            state1 = state_qnode1()

        # From the two generated states, compute the relative entropy
        return qml.math.relative_entropy(state0, state1)

    return evaluate_relative_entropy


def trace_distance(qnode0, qnode1, wires0, wires1):
    r"""
    Compute the trace distance for two :class:`.QNode` returning a :func:`~pennylane.state` (a state can be a state vector
    or a density matrix, depending on the device) acting on quantum systems with the same size.

    .. math::
        T(\rho, \sigma)=\frac12\|\rho-\sigma\|_1
        =\frac12\text{Tr}\left(\sqrt{(\rho-\sigma)^{\dagger}(\rho-\sigma)}\right)

    where :math:`\|\cdot\|_1` is the Schatten :math:`1`-norm.

    The trace distance measures how close two quantum states are. In particular, it upper-bounds
    the probability of distinguishing two quantum states.

    .. warning::

        ``qml.qinfo.trace_distance`` is deprecated and will be removed in v0.40. Instead, use
        :func:`~pennylane.math.trace_distance`.

    Args:
        qnode0 (QNode): A :class:`.QNode` returning a :func:`~pennylane.state`.
        qnode1 (QNode): A :class:`.QNode` returning a :func:`~pennylane.state`.
        wires0 (Sequence[int]): the subsystem of the first QNode.
        wires1 (Sequence[int]): the subsystem of the second QNode.

    Returns:
        func: A function that takes as input the joint arguments of the two QNodes,
        and returns the trace distance between their output states.

    **Example**

    Consider the following QNode:

    .. code-block:: python

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit(param):
            qml.RY(param, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

    The ``qml.qinfo.trace_distance`` transform can be used to compute the trace distance
    between the output states of the QNode:

    >>> trace_distance_circuit = qml.qinfo.trace_distance(circuit, circuit, wires0=[0], wires1=[0])

    The returned function takes two tuples as input, the first being the arguments to the
    first QNode and the second being the arguments to the second QNode:

    >>> x, y = np.array(0.4), np.array(0.6)
    >>> trace_distance_circuit((x,), (y,))
    0.047862689546603415

    This transform is fully differentiable:

    .. code-block:: python

        def wrapper(x, y):
            return trace_distance_circuit((x,), (y,))

    >>> wrapper(x, y)
    0.047862689546603415
    >>> qml.grad(wrapper)(x, y)
    (tensor(-0.19470917, requires_grad=True),
     tensor(0.28232124, requires_grad=True))
    """

    warnings.warn(
        "qml.qinfo.trace_distance is deprecated and will be removed in v0.40. Instead, use "
        "qml.math.trace_distance.",
        qml.PennyLaneDeprecationWarning,
    )

    if len(wires0) != len(wires1):
        raise qml.QuantumFunctionError("The two states must have the same number of wires.")

    # with warnings.catch_warnings():
    warnings.filterwarnings(
        action="ignore",
        message="The qml.qinfo.reduced_dm transform",
        category=qml.PennyLaneDeprecationWarning,
    )
    state_qnode0 = qml.qinfo.reduced_dm(qnode0, wires=wires0)
    state_qnode1 = qml.qinfo.reduced_dm(qnode1, wires=wires1)

    def evaluate_trace_distance(all_args0=None, all_args1=None):
        """Wrapper used for evaluation of the trace distance between two states computed from
        QNodes. It allows giving the args and kwargs to each :class:`.QNode`.

        Args:
            all_args0 (tuple): Tuple containing the arguments (*args, kwargs) of the first :class:`.QNode`.
            all_args1 (tuple): Tuple containing the arguments (*args, kwargs) of the second :class:`.QNode`.

        Returns:
            float: Trace distance between two quantum states
        """
        if not isinstance(all_args0, tuple) and all_args0 is not None:
            all_args0 = (all_args0,)

        if not isinstance(all_args1, tuple) and all_args1 is not None:
            all_args1 = (all_args1,)

        # If no all_args is given, evaluate the QNode without args
        if all_args0 is not None:
            # Handle a dictionary as last argument
            if isinstance(all_args0[-1], dict):
                args0 = all_args0[:-1]
                kwargs0 = all_args0[-1]
            else:
                args0 = all_args0
                kwargs0 = {}
            state0 = state_qnode0(*args0, **kwargs0)
        else:
            # No args
            state0 = state_qnode0()

        # If no all_args is given, evaluate the QNode without args
        if all_args1 is not None:
            # Handle a dictionary as last argument
            if isinstance(all_args1[-1], dict):
                args1 = all_args1[:-1]
                kwargs1 = all_args1[-1]
            else:
                args1 = all_args1
                kwargs1 = {}
            state1 = state_qnode1(*args1, **kwargs1)
        else:
            # No args
            state1 = state_qnode1()

        # From the two generated states, compute the trace distance
        return qml.math.trace_distance(state0, state1)

    return evaluate_trace_distance
