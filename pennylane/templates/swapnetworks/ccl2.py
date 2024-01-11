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
r"""
Contains the TwoLocalSwapNetwork template.
"""

import warnings
import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane.ops import SWAP, FermionicSWAP


class TwoLocalSwapNetwork(Operation):
    r"""Apply two-local gate operations using a canonical 2-complete linear (2-CCL) swap network.

    Args:
        wires (Iterable or Wires): ordered sequence of wires on which the swap network acts
        acquaintances (Callable): callable `func(index, wires, param=None, **kwargs)` that returns
            a two-local operation applied on a pair of logical wires specified by `index` currently
            stored in physical wires provided by `wires` before they are swapped apart.
            Parameters for the operation are specified using `param`, and any additional
            keyword arguments for the callable should be provided using the ``kwargs`` separately
        weights (tensor): weight tensor for the parameterized acquaintances of length
            :math:`N \times (N - 1) / 2`, where `N` is the length of `wires`
        fermionic (bool): If ``True``, qubits are realized as fermionic modes and :class:`~.pennylane.FermionicSWAP` with :math:`\phi=\pi` is used instead of :class:`~.pennylane.SWAP`
        shift (bool): If ``True``, odd-numbered layers begins from the second qubit instead of first one
        **kwargs: additional keyword arguments for `acquaintances`

    Raises:
        ValueError: if inputs do not have the correct format

    **Example**

    .. code-block:: python

        >>> import pennylane as qml
        >>> dev = qml.device('default.qubit', wires=5)
        >>> acquaintances = lambda index, wires, param=None: qml.CNOT(index)
        >>> @qml.qnode(dev)
        ... def swap_network_circuit():
        ...    qml.templates.TwoLocalSwapNetwork(dev.wires, acquaintances, fermionic=True, shift=False)
        ...    return qml.state()
        >>> qml.draw(swap_network_circuit, expansion_strategy='device')()
        0: ─╭●─╭fSWAP(3.14)─────────────────╭●─╭fSWAP(3.14)─────────────────╭●─╭fSWAP(3.14)─┤  State
        1: ─╰X─╰fSWAP(3.14)─╭●─╭fSWAP(3.14)─╰X─╰fSWAP(3.14)─╭●─╭fSWAP(3.14)─╰X─╰fSWAP(3.14)─┤  State
        2: ─╭●─╭fSWAP(3.14)─╰X─╰fSWAP(3.14)─╭●─╭fSWAP(3.14)─╰X─╰fSWAP(3.14)─╭●─╭fSWAP(3.14)─┤  State
        3: ─╰X─╰fSWAP(3.14)─╭●─╭fSWAP(3.14)─╰X─╰fSWAP(3.14)─╭●─╭fSWAP(3.14)─╰X─╰fSWAP(3.14)─┤  State
        4: ─────────────────╰X─╰fSWAP(3.14)─────────────────╰X─╰fSWAP(3.14)─────────────────┤  State

    .. details::
        :title: Usage Details

        More complex acquaintances can be utilized with the template. For example:

        .. code-block:: python

            >>> dev = qml.device('default.qubit', wires=5)
            >>> weights = np.random.random(size=qml.TwoLocalSwapNetwork.shape(len(dev.wires)))
            >>> print(weights)
            tensor([0.20308242, 0.91906199, 0.67988804, 0.81290256, 0.08708985,
                    0.81860084, 0.34448344, 0.05655892, 0.61781612, 0.51829044], requires_grad=True)
            >>> acquaintances = lambda index, wires, param: (qml.CRY(param, wires=index)
            ...                                  if np.abs(wires[0]-wires[1]) else qml.CRZ(param, wires=index))
            >>> @qml.qnode(dev)
            ... def swap_network_circuit():
            ...    qml.templates.TwoLocalSwapNetwork(dev.wires, acquaintances, weights, fermionic=False)
            ...    return qml.state()
            >>> qml.draw(swap_network_circuit, expansion_strategy='device')()
            0: ─╭●────────╭SWAP─────────────────╭●────────╭SWAP─────────────────╭●────────╭SWAP─┤  State
            1: ─╰RY(0.20)─╰SWAP─╭●────────╭SWAP─╰RY(0.09)─╰SWAP─╭●────────╭SWAP─╰RY(0.62)─╰SWAP─┤  State
            2: ─╭●────────╭SWAP─╰RY(0.68)─╰SWAP─╭●────────╭SWAP─╰RY(0.34)─╰SWAP─╭●────────╭SWAP─┤  State
            3: ─╰RY(0.92)─╰SWAP─╭●────────╭SWAP─╰RY(0.82)─╰SWAP─╭●────────╭SWAP─╰RY(0.52)─╰SWAP─┤  State
            4: ─────────────────╰RY(0.81)─╰SWAP─────────────────╰RY(0.06)─╰SWAP─────────────────┤  State

    """

    num_wires = AnyWires
    grad_method = None

    @classmethod
    def _unflatten(cls, data, metadata):
        new_op = cls.__new__(cls)
        new_op._hyperparameters = dict(metadata[1])  # pylint: disable=protected-access
        new_op._weights = data[0]  # pylint: disable=protected-access
        Operation.__init__(new_op, *data, wires=metadata[0])
        return new_op

    def __init__(
        self,
        wires,
        acquaintances=None,
        weights=None,
        fermionic=True,
        shift=False,
        id=None,
        **kwargs,
    ):  # pylint: disable=too-many-arguments
        if len(wires) < 2:
            raise ValueError(f"TwoLocalSwapNetwork requires at least 2 wires, got {len(wires)}")

        if not callable(acquaintances) and acquaintances is not None:
            raise ValueError(
                f"Acquaintances must either be a callable or None, got {acquaintances}"
            )

        if weights is not None and acquaintances is None:
            warnings.warn("Weights are being provided without acquaintances")

        if (
            weights is not None
            and acquaintances is not None
            and qml.math.shape(weights)[0] != int(len(wires) * (len(wires) - 1) / 2)
        ):
            raise ValueError(
                f"Weight tensor must be of length {int(len(wires) * (len(wires) - 1) / 2)}, \
                    got {qml.math.shape(weights)[0]}"
            )

        self._weights = weights
        self._hyperparameters = {
            "acquaintances": acquaintances,
            "fermionic": fermionic,
            "shift": shift,
            **kwargs,
        }

        if acquaintances is not None and self._weights is not None:
            super().__init__(self._weights, wires=wires, id=id)
        else:
            super().__init__(wires=wires, id=id)

    @property
    def num_params(self):
        return (
            1
            if self._hyperparameters["acquaintances"] is not None and self._weights is not None
            else 0
        )

    @staticmethod
    def compute_decomposition(
        weights=None, wires=None, acquaintances=None, fermionic=True, shift=False, **kwargs
    ):  # pylint: disable=arguments-differ too-many-arguments
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.TwoLocalSwapNetwork.decomposition`.

        Args:
            weights (tensor): weight tensor for the parameterized acquaintances of length :math:`N \times (N - 1) / 2`, where `N` is the length of `wires`
            wires (Iterable or Wires): ordered sequence of wires on which the swap network acts
            acquaintances (Callable): callable `func(index, wires, param=None, **kwargs)` that returns a two-local operation, which is applied on a pair of logical wires specified by `index`. This corresponds to applying the operation on physical wires provided by `wires` before any SWAP gates occurred. Parameters for the operation are specified using `param`, and any additional keyword arguments for the callable should be provided using the ``kwargs`` separately
            fermionic (bool): If ``True``, qubits are realized as fermionic modes and :class:`~.pennylane.FermionicSWAP` with :math:`\phi=\pi` is used instead of :class:`~.pennylane.SWAP`
            shift (bool): If ``True``, odd-numbered layers begins from the second qubit instead of first one
            **kwargs: additional keyword arguments for `acquaintances`

        Returns:
            list[.Operator]: decomposition of the operator

        **Example**

        .. code-block:: python

            >>> import pennylane as qml
            >>> dev = qml.device('default.qubit', wires=5)
            >>> acquaintances = lambda index, wires, param=None: qml.CNOT(index)
            >>> qml.TwoLocalSwapNetwork.compute_decomposition(wires=dev.wires,
            ...        acquaintances=acquaintances, fermionic=True, shift=False)
            [CNOT(wires=[0, 1]), FermionicSWAP(3.141592653589793, wires=[0, 1]),
             CNOT(wires=[2, 3]), FermionicSWAP(3.141592653589793, wires=[2, 3]),
             CNOT(wires=[1, 2]), FermionicSWAP(3.141592653589793, wires=[1, 2]),
             CNOT(wires=[3, 4]), FermionicSWAP(3.141592653589793, wires=[3, 4]),
             CNOT(wires=[0, 1]), FermionicSWAP(3.141592653589793, wires=[0, 1]),
             CNOT(wires=[2, 3]), FermionicSWAP(3.141592653589793, wires=[2, 3]),
             CNOT(wires=[1, 2]), FermionicSWAP(3.141592653589793, wires=[1, 2]),
             CNOT(wires=[3, 4]), FermionicSWAP(3.141592653589793, wires=[3, 4]),
             CNOT(wires=[0, 1]), FermionicSWAP(3.141592653589793, wires=[0, 1]),
             CNOT(wires=[2, 3]), FermionicSWAP(3.141592653589793, wires=[2, 3])]
        """

        if wires is None or len(wires) < 2:
            raise ValueError(f"TwoLocalSwapNetwork requires at least 2 wires, got {wires}")

        op_list = []

        wire_order = list(wires).copy()
        itrweights = iter([]) if weights is None or acquaintances is None else iter(weights)
        for layer in range(len(wires)):
            qubit_pairs = [[i, i + 1] for i in range((layer + shift) % 2, len(wires) - 1, 2)]
            for i, j in qubit_pairs:
                qb1, qb2 = wire_order[i], wire_order[j]
                if acquaintances is not None:
                    op_list.append(
                        acquaintances(
                            index=[wires[i], wires[j]],
                            wires=[qb1, qb2],
                            param=next(itrweights, None),
                            **kwargs,
                        )
                    )
                op_list.append(
                    SWAP(wires=[wires[i], wires[j]])
                    if not fermionic
                    else FermionicSWAP(np.pi, wires=[wires[i], wires[j]])
                )
                wire_order[i], wire_order[j] = qb2, qb1

        return op_list

    @staticmethod
    def shape(n_wires):
        r"""Returns the shape of the weight tensor required for using parameterized acquaintances in the template.
        Args:
            n_wires (int): Number of qubits
        Returns:
            tuple[int]: shape
        """

        if n_wires < 2:
            raise ValueError(f"TwoLocalSwapNetwork requires at least 2 wires, got {n_wires}")

        return (int(n_wires * (n_wires - 1) * 0.5),)
