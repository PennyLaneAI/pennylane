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
import pennylane as qml
import pennylane.numpy as np
from pennylane.operation import Operation, AnyWires
from pennylane.ops import SWAP, FermionicSWAP


class TwoLocalSwapNetwork(Operation):
    r"""Apply two-local gate operations using a canonical 2-complete linear (2-CCL) swap network.

    Args:
        wires (Iterable or Wires): ordered sequence of wires on which the swap network acts
        acquaintances (Callable): a callable `func(index, wires, param=None, **kwargs)` that returns a two-local operation being applied on pair of wires specified by `index` that are currently stored in physical qubits or fermionic modes specified by `wires` before they are swapped apart and kwargs accepts any additional keyword argument required for this operation. Note that these kwargs should be provided separately
        weights (tensor): one dimensional tensor to provide weights for the acquaintances of length :math:`N \times (N - 1) / 2`, where `N` is the number of wires and each weight is specified by the `param` argument in the callable given for `acquaintances`. Specify ``None`` if specified acquintances will not be parameterized
        fermionic (bool): If ``True``, qubits are realized as fermionic modes and :class:`~.pennylane.FermionicSWAP` with :math:`\phi=2\pi` is used instead of class:`~.pennylane.SWAP` to build 2-CCL
        shift (bool): If ``True``, odd-numbered layers begins from the second qubit instead of first
        **kwargs: additional keyword arguments for acquaintances

    Raises:
        ValueError: if inputs do not have the correct format

    **Example**

    .. code-block:: python

        >>> import pennylane as qml
        >>> dev = qml.device('default.qubit', wires=5)
        >>> acquaintances = lambda index, wires, param=None: qml.CNOT(index)
        >>> @qml.qnode(dev)
        ... def swap_network_circuit():
        ...    qml.templates.TwoLocalSwapNetwork(None, dev.wires, acquaintances, fermionic=True, shift=False)
        ...    return qml.state()
        >>> qml.draw(swap_network_circuit)()
        0: ─╭●─╭fSWAP───────────╭●─╭fSWAP───────────╭●─╭fSWAP─┤  State
        1: ─╰X─╰fSWAP─╭●─╭fSWAP─╰X─╰fSWAP─╭●─╭fSWAP─╰X─╰fSWAP─┤  State
        2: ─╭●─╭fSWAP─╰X─╰fSWAP─╭●─╭fSWAP─╰X─╰fSWAP─╭●─╭fSWAP─┤  State
        3: ─╰X─╰fSWAP─╭●─╭fSWAP─╰X─╰fSWAP─╭●─╭fSWAP─╰X─╰fSWAP─┤  State
        4: ───────────╰X─╰fSWAP───────────╰X─╰fSWAP───────────┤  State

    .. details::
        :title: Usage Details

        More complex acquaintances can be utilized with the template. For example:

        .. code-block:: python

            >>> dev = qml.device('default.qubit', wires=5)
            >>> weights = np.random.rand(*TwoLocalSwapNetwork.shape(len(dev.wires)))
            >>> print(weights)
            tensor([0.20308242, 0.91906199, 0.67988804, 0.81290256, 0.08708985,
                    0.81860084, 0.34448344, 0.05655892, 0.61781612, 0.51829044], requires_grad=True)
            >>> acquaintances = lambda index, wires, param: (qml.CRY(param, wires=index)
            ...                                  if np.abs(wires[0]-wires[1]) else qml.CRZ(param, wires=index))
            >>> @qml.qnode(dev)
            ... def swap_network_circuit():
            ...    qml.templates.TwoLocalSwapNetwork(weights, dev.wires, acquaintances, fermionic=false)
            ...    return qml.state()
            >>> qml.draw(swap_network_circuit)()
            0: ─╭●────────╭SWAP─────────────────╭●────────╭SWAP─────────────────╭●────────╭SWAP─┤  State
            1: ─╰RY(0.20)─╰SWAP─╭●────────╭SWAP─╰RY(0.09)─╰SWAP─╭●────────╭SWAP─╰RY(0.62)─╰SWAP─┤  State
            2: ─╭●────────╭SWAP─╰RY(0.68)─╰SWAP─╭●────────╭SWAP─╰RY(0.34)─╰SWAP─╭●────────╭SWAP─┤  State
            3: ─╰RY(0.92)─╰SWAP─╭●────────╭SWAP─╰RY(0.82)─╰SWAP─╭●────────╭SWAP─╰RY(0.52)─╰SWAP─┤  State
            4: ─────────────────╰RY(0.81)─╰SWAP─────────────────╰RY(0.06)─╰SWAP─────────────────┤  State

    """

    num_wires = AnyWires
    grad_method = None

    def __init__(
        self,
        wires,
        acquaintances=None,
        weights=None,
        fermionic=True,
        shift=False,
        do_queue=True,
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
            and qml.math.shape(weights) != self.shape(len(wires))
        ):
            raise ValueError(
                f"Weight tensor must be of size {self.shape(len(wires))}, got {qml.math.shape(weights)}"
            )

        self._weights = weights
        self._hyperparameters = {
            "acquaintances": acquaintances,
            "fermionic": fermionic,
            "shift": shift,
            **kwargs,
        }

        if acquaintances is not None and self._weights is not None:
            super().__init__(self._weights, wires=wires, do_queue=do_queue, id=id)
        else:
            super().__init__(wires=wires, do_queue=do_queue, id=id)

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
            weights (tensor): one dimensional tensor to provide weights for the acquaintances of length :math:`N \times (N - 1) / 2`, where `N` is the number of wires and each weight is specified by the `param` argument in the callable given for `acquaintances`. Specify ``None`` if provided acquintances will not be parameterized
            wires (Iterable or Wires): ordered sequence of wires on which the swap network acts
            acquaintances (Callable): a callable `func(index, wires, param=None, **kwargs)` that returns a two-local operation being applied on pair of wires specified by `index` that are currently stored in physical qubits or fermionic modes specified by `wires` before they are swapped apart and kwargs accepts any additional keyword argument required for this operation. Note that these kwargs should be provided separately
            fermionic (bool): If ``True``, qubits are realized as fermionic modes and :class:`~.pennylane.FermionicSWAP` with :math:`\phi=2\pi` is used instead of class:`~.pennylane.SWAP` to build 2-CCL
            shift (bool): If ``True``, odd-numbered layers begins from the second qubit instead of first
            **kwargs: additional keyword arguments for acquaintances

        Returns:
            list[.Operator]: decomposition of the operator
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
                            param=next(itrweights, 0.0),
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
