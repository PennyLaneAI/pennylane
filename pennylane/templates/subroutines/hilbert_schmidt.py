# Copyright 2022 Xanadu Quantum Technologies Inc.

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
This submodule contains the templates for the Hilbert-Schmidt tests.
"""
import copy
from collections.abc import Iterable

from pennylane.math import is_abstract
from pennylane.operation import Operation, Operator
from pennylane.ops import CNOT, Hadamard, QubitUnitary
from pennylane.queuing import QueuingManager, apply
from pennylane.typing import TensorLike
from pennylane.wires import Wires


class HilbertSchmidt(Operation):
    r"""Create a Hilbert-Schmidt template that can be used to compute the Hilbert-Schmidt Test (HST).

    The HST is a useful quantity to compile a target unitary `U` with an approximate unitary `V`. The HST
    is used as a distance between `U` and `V`. The result of executing the HST is 0 if and only if `V` is equal to
    `U` (up to a global phase). As suggested in [1], we can define a cost function using the Hilbert-Schmidt inner product
    between the unitaries `U` and `V` as follows:

    .. math::
        C_{HST} = 1 - \frac{1}{d^2} \left|Tr(V^{\dagger}U)\right|^2,

    where `d` is the dimension of the space in which the unitaries `U` and `V` act.
    The quantity :math:`\frac{1}{d^2} \left|Tr(V^{\dagger}U)\right|^2` is obtained by executing the Hilbert-Schmidt Test.

    It is equivalent to taking the outcome probability of the state :math:`|0 ... 0\rangle`
    for the following circuit:

    .. figure:: ../../_static/templates/subroutines/hst.png
        :align: center
        :width: 80%
        :target: javascript:void(0);

    It defines our decomposition for the Hilbert-Schmidt Test template.

    Args:
        V (Operator or Iterable[Operator]): The operators that represent the unitary `V`.
        U (Operator or Iterable[Operator]): The operators that represent the unitary `U`.
        id (str or None): Optional identifier for the operation.

    Raises:
        ValueError: ``V`` is not an Operator or an iterable of Operators.
        ValueError: ``U`` is not an Operator or an iterable of Operators.
        ValueError: ``U`` and ``V`` do not have the same number of wires.
        ValueError: Operators in ``U`` must act on distinct wires from those in ``v_wires``.

    **Reference**

    [1] Sumeet Khatri, Ryan LaRose, Alexander Poremba, Lukasz Cincio, Andrew T. Sornborger and Patrick J. Coles
    Quantum-assisted Quantum Compiling.
    `arxiv/1807.00800 <https://arxiv.org/pdf/1807.00800.pdf>`_

    .. seealso:: :class:`~.LocalHilbertSchmidt`

    .. details::
        :title: Usage Details

        Consider that we want to evaluate the Hilbert-Schmidt Test cost between the unitary ``U`` and an approximate
        unitary ``V``. If the approximate unitary has fewer wires than the target unitary, a placeholder identity can be included.
        We need to define some functions where it is possible to use the :class:`~.HilbertSchmidt`
        template. In the example below, the considered unitary is ``Hadamard`` and we try to compute the cost for the approximate
        unitary ``RZ``. For an angle that is equal to ``0`` (``Identity``), we have the maximal cost, which is ``1``.

        .. code-block:: python

            U = qml.Hadamard(0)
            V = qml.RZ(0, wires=1)

            dev = qml.device("default.qubit", wires=2)

            @qml.qnode(dev)
            def hilbert_test(V, U):
                qml.HilbertSchmidt(V, U)
                return qml.probs()

            def cost_hst(V, U):
                return 1 - hilbert_test(V, U)[0]

        Now that the cost function has been defined it can be called as follows:

        >>> cost_hst(V, U)
        np.float64(1.0)

    """

    grad_method = None

    @classmethod
    def _primitive_bind_call(cls, V, U, **kwargs):  # kwarg is id
        # pylint: disable=arguments-differ
        U = (U,) if isinstance(U, Operator) or is_abstract(U) else U
        V = (V,) if isinstance(V, Operator) or is_abstract(V) else V
        num_v_ops = len(V)
        return cls._primitive.bind(*V, *U, num_v_ops=num_v_ops, **kwargs)

    def _flatten(self):
        data = (self.hyperparameters["V"], self.hyperparameters["U"])
        return data, tuple()

    @classmethod
    def _unflatten(cls, data, _) -> "HilbertSchmidt":
        return cls(*data)

    def __init__(
        self,
        V: Operator | Iterable[Operator],
        U: Operator | Iterable[Operator],
        id: str | None = None,
    ) -> None:

        u_ops = (U,) if isinstance(U, Operator) else tuple(U)
        if not all(isinstance(op, Operator) for op in u_ops):
            raise ValueError("The argument 'U' must be an Operator or an iterable of Operators.")
        u_wires = Wires.all_wires([op.wires for op in u_ops])

        v_ops = (V,) if isinstance(V, Operator) else tuple(V)
        if not all(isinstance(op, Operator) for op in v_ops):
            raise ValueError("The argument 'V' must be an Operator or an iterable of Operators.")
        v_wires = Wires.all_wires([op.wires for op in v_ops])

        self._hyperparameters = {
            "U": u_ops,
            "V": v_ops,
        }

        if len(u_wires) != len(v_wires):
            raise ValueError("U and V must have the same number of wires.")

        if len(Wires.shared_wires([u_wires, v_wires])) != 0:
            raise ValueError("Operators in U and V must act on distinct wires.")

        total_wires = Wires(u_wires + v_wires)
        super().__init__(wires=total_wires, id=id)

    def map_wires(self, wire_map: dict):
        raise NotImplementedError("Mapping the wires of HilbertSchmidt is not implemented.")

    @property
    def data(self):
        r"""Flattened list of operator data in this HilbertSchmidt operation."""
        return tuple(datum for op in self._operators for datum in op.data)

    @data.setter
    def data(self, new_data):
        # We need to check if ``new_data`` is empty because ``Operator.__init__()``  will attempt to
        # assign the HilbertSchmidt data to an empty tuple (since no positional arguments are provided).
        if new_data:
            for op in self._operators:
                if op.num_params > 0:
                    op.data = new_data[: op.num_params]
                    new_data = new_data[op.num_params :]

    def __copy__(self):
        # Override Operator.__copy__() to avoid setting the "data" property before the new instance
        # is assigned hyper-parameters since HilbertSchmidt data is derived from the hyper-parameters.
        clone = HilbertSchmidt.__new__(HilbertSchmidt)

        # Ensure the operators in the hyper-parameters are copied instead of aliased.
        clone._hyperparameters = {
            "U": list(map(copy.copy, self._hyperparameters["U"])),
            "V": list(map(copy.copy, self._hyperparameters["V"])),
        }

        for attr, value in vars(self).items():
            if attr != "_hyperparameters":
                setattr(clone, attr, value)

        return clone

    @property
    def _operators(self) -> list[Operator]:
        """Flattened list of operators that compose this HilbertSchmidt operation."""
        return [*self._hyperparameters["V"], *self._hyperparameters["U"]]

    def queue(self, context=QueuingManager) -> "HilbertSchmidt":
        for op in self._hyperparameters["V"]:
            context.remove(op)
        for op in self._hyperparameters["U"]:
            context.remove(op)
        context.append(self)
        return self

    @staticmethod
    def compute_decomposition(
        *params: TensorLike,
        wires: int | Iterable[int | str] | Wires,
        U: Operator | Iterable[Operator],
        V: Operator | Iterable[Operator],
    ) -> list[Operator]:
        # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators."""

        u_ops = (U,) if isinstance(U, Operator) else tuple(U)
        v_ops = (V,) if isinstance(V, Operator) else tuple(V)
        u_wires = Wires.all_wires([op.wires for op in u_ops])
        v_wires = Wires.all_wires([op.wires for op in v_ops])

        n_wires = len(u_wires + v_wires)
        first_range = range(n_wires // 2)
        second_range = range(n_wires // 2, n_wires)

        # Hadamard first layer
        decomp_ops = [Hadamard(wires[i]) for i in first_range]
        # CNOT first layer
        decomp_ops.extend(
            CNOT(wires=[wires[i], wires[j]]) for i, j in zip(first_range, second_range)
        )

        # Unitary U
        for op_u in u_ops:
            # The operation has been defined outside of this function, to queue it we call qml.apply.
            if QueuingManager.recording():
                apply(op_u)
            decomp_ops.append(op_u)

        # Unitary V conjugate
        # Since we don't currently have an easy way to apply the complex conjugate of a tape, we manually
        # apply the complex conjugate of each operator in the V tape and append it to the decomposition
        # using the QubitUnitary operation.
        for op_v in v_ops:
            mat = op_v.matrix().conjugate()
            decomp_ops.append(QubitUnitary(mat, wires=op_v.wires))

        # CNOT second layer
        decomp_ops.extend(
            CNOT(wires=[wires[i], wires[j]])
            for i, j in zip(reversed(first_range), reversed(second_range))
        )
        # Hadamard second layer
        decomp_ops.extend(Hadamard(wires[i]) for i in first_range)
        return decomp_ops


# pylint: disable=protected-access
if HilbertSchmidt._primitive is not None:

    @HilbertSchmidt._primitive.def_impl
    def _(*ops, num_v_ops, **kwargs):
        V = ops[:num_v_ops]
        U = ops[num_v_ops:]
        return type.__call__(HilbertSchmidt, V, U, **kwargs)


class LocalHilbertSchmidt(HilbertSchmidt):
    r"""Create a Local Hilbert-Schmidt template that can be used to compute the Local Hilbert-Schmidt Test (LHST).

    The result of the LHST is a useful quantity for compiling a unitary `U` with an approximate unitary `V`. The
    LHST is used as a distance between `U` and `V`. It is similar to the Hilbert-Schmidt test, but the measurement is
    made only on one qubit at the end of the circuit. The LHST cost is always smaller than the HST cost and is useful
    for large unitaries.

    .. figure:: ../../_static/templates/subroutines/lhst.png
        :align: center
        :width: 80%
        :target: javascript:void(0);

    Args:
        V (Operator or Iterable[Operator]): The operators that represent the approximate compiled unitary `V`.
        U (Operator or Iterable[Operator]): The operators that represent the unitary `U`.
        id (str or None): Optional identifier for the operation.

    Raises:
        ValueError: ``V`` is not an Operator or an iterable of Operators.
        ValueError: ``U`` is not an Operator or an iterable of Operators.
        ValueError: ``U`` and ``V`` do not have the same number of wires.
        ValueError: Operators in ``U`` must act on distinct wires from those in ``v_wires``.

    **Reference**

    [1] Sumeet Khatri, Ryan LaRose, Alexander Poremba, Lukasz Cincio, Andrew T. Sornborger and Patrick J. Coles
    Quantum-assisted Quantum Compiling.
    `arxiv/1807.00800 <https://arxiv.org/pdf/1807.00800.pdf>`_

    .. seealso:: :class:`~.HilbertSchmidt`

    .. details::
        :title: Usage Details

        Consider that we want to evaluate the Local Hilbert-Schmidt Test cost between the unitary ``U`` and an
        approximate unitary ``V``. We need to define some functions where it is possible to use the
        :class:`~.LocalHilbertSchmidt` template. Here the considered unitary is ``CZ`` and we try to compute the
        cost for the approximate unitary.

        .. code-block:: python

            import numpy as np

            params = [3 * np.pi / 2, 3 * np.pi / 2, np.pi / 2]

            U = qml.CZ(wires=(0, 1))

            V = [qml.RZ(params[0], wires=2),
                qml.RZ(params[1], wires=3),
                qml.CNOT(wires=[2, 3]),
                qml.RZ(params[2], wires=3),
                qml.CNOT(wires=[2, 3])]

            dev = qml.device("default.qubit", wires=4)

            @qml.qnode(dev)
            def local_hilbert_test(V, U):
                qml.LocalHilbertSchmidt(V, U)
                return qml.probs()

            def cost_lhst(V, U):
                return 1 - local_hilbert_test(V, U)[0]

        Now that the cost function has been defined it can be called for specific parameters:

        >>> cost_lhst(V, U)
        np.float64(0.5...)
    """

    @staticmethod
    def compute_decomposition(
        *params: TensorLike,
        wires: int | Iterable[int | str] | Wires,
        U: Operator | Iterable[Operator],
        V: Operator | Iterable[Operator],
    ) -> list[Operator]:
        r"""Representation of the operator as a product of other operators (static method)."""

        u_ops = (U,) if isinstance(U, Operator) else tuple(U)
        v_ops = (V,) if isinstance(V, Operator) else tuple(V)
        u_wires = Wires.all_wires([op.wires for op in u_ops])
        v_wires = Wires.all_wires([op.wires for op in v_ops])

        n_wires = len(u_wires + v_wires)
        first_range = range(n_wires // 2)
        second_range = range(n_wires // 2, n_wires)

        # Hadamard first layer
        decomp_ops = [Hadamard(wires[i]) for i in first_range]
        # CNOT first layer
        decomp_ops.extend(
            CNOT(wires=[wires[i], wires[j]]) for i, j in zip(first_range, second_range)
        )

        # Unitary U
        if QueuingManager.recording():
            decomp_ops.extend(apply(op_u) for op_u in u_ops)
        else:
            decomp_ops.extend(u_ops)

        # Unitary V conjugate
        # Since we don't currently have an easy way to apply the complex conjugate of a tape, we manually
        # apply the complex conjugate of each operation in the V tape and append it to the decomposition
        # using the QubitUnitary operation.
        decomp_ops.extend(
            QubitUnitary(op_v.matrix().conjugate(), wires=op_v.wires) for op_v in v_ops
        )
        # Single qubit measurement
        decomp_ops.extend((CNOT(wires=[wires[0], wires[n_wires // 2]]), Hadamard(wires[0])))

        return decomp_ops

    def __copy__(self):
        clone = LocalHilbertSchmidt.__new__(LocalHilbertSchmidt)
        clone._hyperparameters = {
            "U": list(map(copy.copy, self._hyperparameters["U"])),
            "V": list(map(copy.copy, self._hyperparameters["V"])),
        }
        for attr, value in vars(self).items():
            if attr != "_hyperparameters":
                setattr(clone, attr, value)
        return clone


# pylint: disable=protected-access
if LocalHilbertSchmidt._primitive is not None:

    @LocalHilbertSchmidt._primitive.def_impl
    def _(*ops, num_v_ops, **kwargs):
        V = ops[:num_v_ops]
        U = ops[num_v_ops:]
        return type.__call__(LocalHilbertSchmidt, V, U, **kwargs)
