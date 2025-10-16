# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Contains the PhaseAdder template.
"""

import numpy as np

from pennylane import math
from pennylane.control_flow import for_loop
from pennylane.decomposition import (
    add_decomps,
    adjoint_resource_rep,
    register_resources,
    resource_rep,
)
from pennylane.operation import Operation
from pennylane.ops import CNOT, ControlledPhaseShift, PhaseShift, X, adjoint, ctrl
from pennylane.templates.subroutines.qft import QFT
from pennylane.wires import Wires, WiresLike


def _add_k_fourier(k, wires: WiresLike):
    """Adds k in the Fourier basis"""
    op_list = []
    for j, wire in enumerate(wires):
        op_list.append(PhaseShift(k * np.pi / (2**j), wires=wire))
    return op_list


class PhaseAdder(Operation):
    r"""Performs the in-place modular phase addition operation.

    This operator performs the modular addition by an integer :math:`k` modulo :math:`mod` in the
    Fourier basis:

    .. math::

        \text{PhaseAdder}(k,mod) |\phi (x) \rangle = |\phi (x+k \; \text{mod} \; mod) \rangle,

    where :math:`|\phi (x) \rangle` represents the :math:`| x \rangle` state in the Fourier basis,

    .. math::

        \text{QFT} |x \rangle = |\phi (x) \rangle.

    The implementation is based on the quantum Fourier transform method presented in
    `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_.

    .. note::

        To obtain the correct result, :math:`x` must be smaller than :math:`mod`. Also, when
        :math:`mod \neq 2^{\text{len(x_wires)}}`, :math:`x` must satisfy :math:`x < 2^{\text{len(x_wires)}-1}`,
        which means that one extra wire in ``x_wires`` is required.

    .. seealso:: :class:`~.QFT` and :class:`~.Adder`.

    Args:
        k (int): the number that needs to be added
        x_wires (Sequence[int]): the wires the operation acts on. The number of wires must be enough
            for a binary representation of the value being targeted, :math:`x`. In some cases an additional
            wire is needed, see usage details below. The number of wires also limits the maximum
            value for ``mod``.
        mod (int): the modulo for performing the addition. If not provided, it will be set to its maximum value, :math:`2^{\text{len(x_wires)}}`.
        work_wire (Sequence[int] or int): the auxiliary wire to use for the addition. Optional
            when ``mod`` is :math:`2^{\text{len(x_wires)}}`. Defaults to empty tuple. The work wire must
            be in the :math:`|0\rangle` state.

    **Example**

    This example computes the sum of two integers :math:`x=8` and :math:`k=5` modulo :math:`mod=15`.

    .. code-block:: python

        x = 8
        k = 5
        mod = 15

        x_wires =[0,1,2,3]
        work_wire=[5]

        dev = qml.device("default.qubit")

        @qml.qnode(dev, shots=1)
        def circuit():
            qml.BasisEmbedding(x, wires=x_wires)
            qml.QFT(wires=x_wires)
            qml.PhaseAdder(k, x_wires, mod, work_wire)
            qml.adjoint(qml.QFT)(wires=x_wires)
            return qml.sample(wires=x_wires)

    >>> print(circuit())
    [[1 1 0 1]]

    The result, :math:`[[1 1 0 1]]`, is the binary representation of
    :math:`8 + 5  \; \text{modulo} \; 15 = 13`.

    .. details::
        :title: Usage Details

        This template takes as input two different sets of wires.

        The first one is ``x_wires``, used to encode the integer :math:`x < \text{mod}` in the Fourier basis.
        To represent :math:`x`, at least :math:`\lceil \log_2(x) \rceil` wires are needed.
        After the modular addition, the result can be as large as :math:`\text{mod} - 1`,
        requiring at least :math:`\lceil \log_2(\text{mod}) \rceil` wires. Since :math:`x < \text{mod}`, a length of
        :math:`\lceil \log_2(\text{mod}) \rceil` is sufficient for ``x_wires`` to cover all possible inputs and
        outputs when :math:`mod = 2^{\text{len(x_wires)}}`.
        An exception occurs when :math:`mod \neq 2^{\text{len(x_wires)}}`. In that case one extra wire in ``x_wires`` will be needed to correctly perform the phase
        addition operation.

        The second set of wires is ``work_wire`` which consist of the auxiliary qubit used to perform the modular phase addition operation.

        - If :math:`mod = 2^{\text{len(x_wires)}}`, there will be no need for ``work_wire``, hence ``work_wire=()``. This is the case by default.

        - If :math:`mod \neq 2^{\text{len(x_wires)}}`, one ``work_wire`` has to be provided.

        Note that the ``PhaseAdder`` template allows us to perform modular addition in the Fourier basis. However if one just wants to perform standard addition (with no modulo),
        that would be equivalent to setting the modulo :math:`mod` to a large enough value to ensure that :math:`x+k < mod`.
    """

    grad_method = None

    resource_keys = {"num_x_wires", "mod"}

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(self, k, x_wires: WiresLike, mod=None, work_wire: WiresLike = (), id=None):

        work_wire = Wires(() if work_wire is None else work_wire)
        x_wires = Wires(x_wires)

        num_work_wires = len(work_wire)

        if not math.is_abstract(mod):
            if mod is None:
                mod = 2 ** len(x_wires)
            elif mod != 2 ** len(x_wires) and num_work_wires != 1:
                raise ValueError(
                    f"If mod is not 2^{len(x_wires)}, one work wire should be provided."
                )
            if not isinstance(k, int) or not isinstance(mod, int):
                raise ValueError("Both k and mod must be integers")
            if mod > 2 ** len(x_wires):
                raise ValueError(
                    "PhaseAdder must have enough x_wires to represent mod. The maximum mod "
                    f"with len(x_wires)={len(x_wires)} is {2 ** len(x_wires)}, but received {mod}."
                )
            if num_work_wires != 0:
                if any(wire in work_wire for wire in x_wires):
                    raise ValueError(
                        "None of the wires in work_wire should be included in x_wires."
                    )

        self.hyperparameters["k"] = k % mod
        self.hyperparameters["mod"] = mod
        self.hyperparameters["work_wire"] = work_wire
        self.hyperparameters["x_wires"] = x_wires
        super().__init__(wires=x_wires, id=id)

    @property
    def resource_params(self) -> dict:
        return {
            "num_x_wires": len(self.hyperparameters["x_wires"]),
            "mod": self.hyperparameters["mod"],
        }

    @property
    def num_params(self):
        return 0

    def _flatten(self):
        metadata = tuple((key, value) for key, value in self.hyperparameters.items())
        return tuple(), metadata

    @classmethod
    def _unflatten(cls, data, metadata):
        hyperparams_dict = dict(metadata)
        return cls(**hyperparams_dict)

    def map_wires(self, wire_map: dict):
        new_dict = {
            key: ([wire_map.get(w, w) for w in self.hyperparameters[key]])
            for key in ["x_wires", "work_wire"]
        }

        return PhaseAdder(
            self.hyperparameters["k"],
            new_dict["x_wires"],
            self.hyperparameters["mod"],
            new_dict["work_wire"],
        )

    def decomposition(self):
        return self.compute_decomposition(**self.hyperparameters)

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @staticmethod
    def compute_decomposition(
        k, x_wires: WiresLike, mod, work_wire: WiresLike
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        Args:
            k (int): the number that needs to be added
            x_wires (Sequence[int]): the wires the operation acts on. The number of wires must be enough
                for a binary representation of the value being targeted, :math:`x`. In some cases an additional
                wire is needed, see usage details below. The number of wires also limits the maximum
                value for `mod`.
            mod (int): the modulo for performing the addition. If not provided, it will be set to its maximum value, :math:`2^{\text{len(x_wires)}}`.
            work_wire (Sequence[int]): the auxiliary wire to use for the addition. Optional
                when `mod` is :math:`2^{len(x\_wires)}`.
        Returns:
            list[.Operator]: Decomposition of the operator

        **Example**

        >>> qml.PhaseAdder.compute_decomposition(k = 2, x_wires = [0, 1, 2], mod = 8, work_wire = ())
        [PhaseShift(6.28..., wires=[0]), PhaseShift(3.141..., wires=[1]), PhaseShift(1.57..., wires=[2])]
        """
        op_list = []

        if mod == 2 ** len(x_wires):
            op_list.extend(_add_k_fourier(k, x_wires))
        else:
            aux_k = x_wires[0]
            op_list.extend(_add_k_fourier(k, x_wires))

            for op in reversed(_add_k_fourier(mod, x_wires)):
                op_list.append(adjoint(op))

            op_list.append(adjoint(QFT)(wires=x_wires))
            op_list.append(ctrl(X(work_wire), control=aux_k, control_values=1))
            op_list.append(QFT(wires=x_wires))
            op_list.extend(ctrl(op, control=work_wire) for op in _add_k_fourier(mod, x_wires))

            for op in reversed(_add_k_fourier(k, x_wires)):
                op_list.append(adjoint(op))

            op_list.append(adjoint(QFT)(wires=x_wires))
            op_list.append(ctrl(X(work_wire), control=aux_k, control_values=0))
            op_list.append(QFT(wires=x_wires))
            op_list.extend(_add_k_fourier(k, x_wires))

        return op_list


def _phase_adder_decomposition_resources(num_x_wires, mod) -> dict:

    if mod == 2**num_x_wires:
        return {PhaseShift: num_x_wires}

    return {
        PhaseShift: 2 * num_x_wires,
        adjoint_resource_rep(PhaseShift, {}): 2 * num_x_wires,
        resource_rep(QFT, num_wires=num_x_wires): 2,
        adjoint_resource_rep(QFT, {"num_wires": num_x_wires}): 2,
        CNOT: 2,
        X: 2,
        ControlledPhaseShift: num_x_wires,
    }


# pylint: disable=no-value-for-parameter
@register_resources(_phase_adder_decomposition_resources)
def _phase_adder_decomposition(k, x_wires: WiresLike, mod, work_wire, **__):

    n_wires = len(x_wires)

    @for_loop(n_wires)
    def _add_k_fourier_loop(i, _k):
        PhaseShift(_k * np.pi / (2**i), wires=x_wires[i])
        return _k

    if mod == 2**n_wires:
        _add_k_fourier_loop(k)
        return

    aux_k = x_wires[0]
    _add_k_fourier_loop(k)
    adjoint(_add_k_fourier_loop)(mod)
    adjoint(QFT)(wires=x_wires)
    CNOT(wires=[aux_k, work_wire[0]])
    QFT(wires=x_wires)
    ctrl(_add_k_fourier_loop, control=work_wire)(mod)
    adjoint(_add_k_fourier_loop)(k)
    adjoint(QFT)(wires=x_wires)
    X(aux_k)
    CNOT(wires=[aux_k, work_wire[0]])
    X(aux_k)
    QFT(wires=x_wires)
    _add_k_fourier_loop(k)


add_decomps(PhaseAdder, _phase_adder_decomposition)
