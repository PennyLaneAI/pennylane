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
"""Code for resource estimation"""
import inspect

import pennylane as qml


def _get_absolute_import_path(fn):
    return f"{inspect.getmodule(fn).__name__}.{fn.__name__}"


def specs(qnode, max_expansion=None, expansion_strategy=None):
    """Resource information about a quantum circuit.

    This transform converts a QNode into a callable that provides resource information
    about the circuit.

    Args:
        qnode (.QNode): the QNode to calculate the specifications for

    Keyword Args:
        max_expansion (int): The number of times the internal circuit should be expanded when
            calculating the specification. Defaults to ``qnode.max_expansion``.
        expansion_strategy (str): The strategy to use when circuit expansions or decompositions
            are required.

            - ``gradient``: The QNode will attempt to decompose
              the internal circuit such that all circuit operations are supported by the gradient
              method.

            - ``device``: The QNode will attempt to decompose the internal circuit
              such that all circuit operations are natively supported by the device.

    Returns:
        A function that has the same argument signature as ``qnode``. This function
        returns a dictionary of information about qnode structure.

    **Example**

    .. code-block:: python3

        x = np.array([0.1, 0.2])
        hamiltonian = qml.dot([1.0, 0.5], [qml.X(0), qml.Y(0)])

        dev = qml.device('default.qubit', wires=2)
        @qml.qnode(dev, diff_method="parameter-shift", shifts=np.pi / 4)
        def circuit(x, add_ry=True):
            qml.RX(x[0], wires=0)
            qml.CNOT(wires=(0,1))
            qml.TrotterProduct(hamiltonian, time=1.0, n=4, order=2)
            if add_ry:
                qml.RY(x[1], wires=1)
            qml.TrotterProduct(hamiltonian, time=1.0, n=4, order=4)
            return qml.probs(wires=(0,1))

    >>> qml.specs(circuit)(x, add_ry=False)
    {'resources': Resources(num_wires=2, num_gates=4, gate_types=defaultdict(<class 'int'>, {'RX': 1, 'CNOT': 1, 'TrotterPro
    duct': 2}}), gate_sizes=defaultdict(<class 'int'>, {1: 3, 2: 1}), depth=4, shots=Shots(total_shots=None, shot_vector=())),
    'errors': {'SpectralNormError': SpectralNormError(0.42998560822421455)},
    'num_observables': 1,
    'num_diagonalizing_gates': 0,
    'num_trainable_params': 1,
    'num_device_wires': 2,
    'device_name': 'default.qubit',
    'expansion_strategy': 'gradient',
    'gradient_options': {'shifts': 0.7853981633974483},
    'interface': 'auto',
    'diff_method': 'parameter-shift',
    'gradient_fn': 'pennylane.gradients.parameter_shift.param_shift',
    'num_gradient_executions': 2}

    """

    def specs_qnode(*args, **kwargs):
        """Returns information on the structure and makeup of provided QNode.

        Dictionary keys:
            * ``"num_operations"`` number of operations in the qnode
            * ``"num_observables"`` number of observables in the qnode
            * ``"num_diagonalizing_gates"`` number of diagonalizing gates required for execution of the qnode
            * ``"resources"``: a :class:`~.resource.Resources` object containing resource quantities used by the qnode
            * ``"errors"``: combined algorithmic errors from the quantum operations executed by the qnode
            * ``"num_used_wires"``: number of wires used by the circuit
            * ``"num_device_wires"``: number of wires in device
            * ``"depth"``: longest path in directed acyclic graph representation
            * ``"device_name"``: name of QNode device
            * ``"expansion_strategy"``: string specifying method for decomposing operations in the circuit
            * ``"gradient_options"``: additional configurations for gradient computations
            * ``"interface"``: autodiff framework to dispatch to for the qnode execution
            * ``"diff_method"``: a string specifying the differntiation method
            * ``"gradient_fn"``: executable to compute the gradient of the qnode

        Potential Additional Information:
            * ``"num_trainable_params"``: number of individual scalars that are trainable
            * ``"num_gradient_executions"``: number of times circuit will execute when
                    calculating the derivative

        Returns:
            dict[str, Union[defaultdict,int]]: dictionaries that contain QNode specifications
        """
        initial_max_expansion = qnode.max_expansion
        initial_expansion_strategy = getattr(qnode, "expansion_strategy", None)

        try:
            qnode.max_expansion = initial_max_expansion if max_expansion is None else max_expansion
            qnode.expansion_strategy = expansion_strategy or initial_expansion_strategy
            qnode.construct(args, kwargs)
        finally:
            qnode.max_expansion = initial_max_expansion
            qnode.expansion_strategy = initial_expansion_strategy

        info = qnode.qtape.specs.copy()

        info["num_device_wires"] = (
            len(qnode.tape.wires)
            if isinstance(qnode.device, qml.devices.Device)
            else len(qnode.device.wires)
        )
        info["device_name"] = getattr(qnode.device, "short_name", qnode.device.name)
        info["expansion_strategy"] = qnode.expansion_strategy
        info["gradient_options"] = qnode.gradient_kwargs
        info["interface"] = qnode.interface
        info["diff_method"] = (
            _get_absolute_import_path(qnode.diff_method)
            if callable(qnode.diff_method)
            else qnode.diff_method
        )

        if isinstance(qnode.gradient_fn, qml.transforms.core.TransformDispatcher):
            info["gradient_fn"] = _get_absolute_import_path(qnode.gradient_fn)

            try:
                info["num_gradient_executions"] = len(qnode.gradient_fn(qnode.qtape)[0])
            except Exception as e:  # pylint: disable=broad-except
                # In the case of a broad exception, we don't want the `qml.specs` transform
                # to fail. Instead, we simply indicate that the number of gradient executions
                # is not supported for the reason specified.
                info["num_gradient_executions"] = f"NotSupported: {str(e)}"
        else:
            info["gradient_fn"] = qnode.gradient_fn

        return info

    return specs_qnode
