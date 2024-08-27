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

import pennylane as qml
from pennylane.operation import Operation


def _add_k_fourier(k, wires):
    """Adds k in the Fourier basis"""
    op_list = []
    for j, wire in enumerate(wires):
        op_list.append(qml.PhaseShift(k * np.pi / (2**j), wires=wire))
    return op_list


class PhaseAdder(Operation):
    r"""Performs the in-place modular phase addition operation.

    This operator performs the modular addition by an integer :math:`k` modulo :math:`mod` in the
    Fourier basis:

    .. math::

        \text{PhaseAdder}(k,mod) |\phi (x) \rangle = |\phi (x+k \; (mod)) \rangle,

    where :math:`|\phi (x) \rangle` represents the :math:`| x \rangle` state in the Fourier basis,

    .. math::

        \text{QFT} |x \rangle = |\phi (x) \rangle.
    
    This operation can be represented in a quantum circuit as:

    .. figure:: ../../_static/templates/subroutines/arithmetic/phaseadder.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    The implementation is based on the quantum Fourier transform method presented in
    `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_.

    .. note::

        Note that :math:`x` must be smaller than :math:`mod` to get the correct result. Also, when
        :math:`mod \neq 2^{\text{len(x_wires)}}` we need :math:`x < 2^{\text{len(x_wires)}}/2`,
        which means that we need one extra wire in ``x_wires``.

    .. seealso:: :class:`~.QFT` and :class:`~.Adder`.

    Args:
        k (int): the number that needs to be added
        x_wires (Sequence[int]): the wires the operation acts on
        mod (int): the modulus for performing the addition, default value is :math:`2^{\text{len(x_wires)}}`
        work_wire (Sequence[int]): the auxiliary wire to be used for performing the addition
            when :math:`mod \neq 2^{\text{len(x_wires)}}`

    **Example**

    This example computes the sum of two integers :math:`x=8` and :math:`k=5` modulo :math:`mod=15`.

    .. code-block::

        x = 8
        k = 5
        mod = 15

        x_wires =[0,1,2,3]
        work_wire=[5]

        dev = qml.device("default.qubit", shots=1)
        @qml.qnode(dev)
        def circuit():
            qml.BasisEmbedding(x, wires=x_wires)
            qml.QFT(wires=x_wires)
            qml.PhaseAdder(k, x_wires, mod, work_wire)
            qml.adjoint(qml.QFT)(wires=x_wires)
            return qml.sample(wires=x_wires)

    .. code-block:: pycon

        >>> print(circuit())
        [1 1 0 1]

    The result, :math:`[1 1 0 1]`, is the ket representation of
    :math:`8 + 5  \, \text{modulo} \, 15 = 13`.

    .. details::
        :title: Usage Details

        This template takes as input two different sets of wires. 
        
        The first one is ``x_wires`` which is used
        to encode the integer :math:`x < mod` in the Fourier basis. After performing the modular phase addition operation the result integer
        encoded in the Fourier basis can be as large as :math:`mod-1`. Therefore, we need at least 
        :math:`\lceil \log_2(x)\rceil` ``x_wires`` to represent :math:`x` and at least :math:`\lceil \log_2(mod)\rceil` ``x_wires`` 
        to represent all the possible results. Since :math:`x < mod` by definition, we just need at least :math:`\lceil \log_2(mod)\rceil` ``x_wires``.
        An exception occur when :math:`mod \neq 2^{\text{len(x_wires)}}`. In that case we will need one extra wire in ``x_wires`` to correctly perform the phase 
        addition operation. 

        The second set of wires is ``work_wire`` which consist of the auxiliary qubit used to perform the modular phase addition operation. 

        If :math:`mod = 2^{\text{len(x_wires)}}`, there will be no need for ``work_wire``, hence ``work_wire=None``. This is the case by default.

        If :math:`mod \neq 2^{\text{len(x_wires)}}`, one ``work_wire`` has to be provided.

        Note that the PhaseAdder template allows us to perform modular addition in the Fourier basis. However if one just want to perform standard addition (with no modulo), the modulo 
        :math:`mod` has to be set large enough to ensure that :math:`x+k < mod`.
    """

    grad_method = None

    def __init__(
        self, k, x_wires, mod=None, work_wire=None, id=None
    ):  # pylint: disable=too-many-arguments

        x_wires = qml.wires.Wires(x_wires)
        if mod is None:
            mod = 2 ** len(x_wires)
        elif work_wire is None and mod != 2 ** len(x_wires):
            raise ValueError(f"If mod is not 2^{len(x_wires)}, one work wire should be provided.")
        if not isinstance(k, int) or not isinstance(mod, int):
            raise ValueError("Both k and mod must be integers")
        if mod > 2 ** len(x_wires):
            raise ValueError("PhaseAdder must have enough x_wires to represent mod.")
        if work_wire is not None:
            if any(wire in work_wire for wire in x_wires):
                raise ValueError("None of the wires in work_wire should be included in x_wires.")

        self.hyperparameters["k"] = k % mod
        self.hyperparameters["mod"] = mod
        self.hyperparameters["work_wire"] = qml.wires.Wires(work_wire)
        self.hyperparameters["x_wires"] = x_wires
        all_wires = qml.wires.Wires(x_wires) + qml.wires.Wires(work_wire)
        super().__init__(wires=all_wires, id=id)

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
            key: [wire_map.get(w, w) for w in self.hyperparameters[key]]
            for key in ["x_wires", "work_wire"]
        }

        return PhaseAdder(
            self.hyperparameters["k"],
            new_dict["x_wires"],
            self.hyperparameters["mod"],
            new_dict["work_wire"],
        )

    def decomposition(self):  # pylint: disable=arguments-differ
        return self.compute_decomposition(**self.hyperparameters)

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @staticmethod
    def compute_decomposition(k, x_wires, mod, work_wire):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        Args:
            k (int): the number that needs to be added
            x_wires (Sequence[int]): the wires the operation acts on
            mod (int): the modulus for performing the addition, default value is :math:`2^{\text{len(x_wires)}}`
            work_wire (Sequence[int]): the auxiliary wire to be used for performing the addition
                when :math:`mod \neq 2^{\text{len(x_wires)}}`
        Returns:
            list[.Operator]: Decomposition of the operator

        **Example**

        >>> qml.PhaseAdder.compute_decomposition(k = 2, x_wires = [0, 1, 2], mod = 8, work_wire = None)
        [PhaseShift(6.283185307179586, wires=[1]),
        PhaseShift(3.141592653589793, wires=[2]),
        PhaseShift(1.5707963267948966, wires=[3])]
        """
        op_list = []

        if mod == 2 ** len(x_wires):
            op_list.extend(_add_k_fourier(k, x_wires))
        else:
            aux_k = x_wires[0]
            op_list.extend(_add_k_fourier(k, x_wires))
            op_list.extend(qml.adjoint(_add_k_fourier)(mod, x_wires))
            op_list.append(qml.adjoint(qml.QFT)(wires=x_wires))
            op_list.append(qml.ctrl(qml.PauliX(work_wire), control=aux_k, control_values=1))
            op_list.append(qml.QFT(wires=x_wires))
            op_list.extend(qml.ctrl(op, control=work_wire) for op in _add_k_fourier(mod, x_wires))
            op_list.extend(qml.adjoint(_add_k_fourier)(k, x_wires))
            op_list.append(qml.adjoint(qml.QFT)(wires=x_wires))
            op_list.append(qml.ctrl(qml.PauliX(work_wire), control=aux_k, control_values=0))
            op_list.append(qml.QFT(wires=x_wires))
            op_list.extend(_add_k_fourier(k, x_wires))

        return op_list
