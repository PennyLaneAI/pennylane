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
from collections.abc import Iterable

# pylint: disable-msg=too-many-arguments
import pennylane as qml
from pennylane.exceptions import QuantumFunctionError
from pennylane.operation import Operation
from pennylane.typing import Callable, TensorLike


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
        *params (array): Parameters for the quantum function `V`.
        v_function (callable): Quantum function that represents the approximate compiled unitary `V`.
        v_wires (int or Iterable[Number, str]]): The wire(s) on which the approximate compiled unitary acts.
        u (Operation or Iterable[Operation]): The operations that represent the unitary `U`.

    Raises:
        QuantumFunctionError: The argument ``u`` is not an Operator or an iterable of Operators.
        QuantumFunctionError: ``v_function`` is not a valid quantum function.
        QuantumFunctionError: ``U`` and ``V`` do not have the same number of wires.
        QuantumFunctionError: The wires ``v_wires`` are a subset of ``V`` wires.
        QuantumFunctionError: Operations in ``u`` must act on distinct wires from those in ``v_wires``.

    **Reference**

    [1] Sumeet Khatri, Ryan LaRose, Alexander Poremba, Lukasz Cincio, Andrew T. Sornborger and Patrick J. Coles
    Quantum-assisted Quantum Compiling.
    `arxiv/1807.00800 <https://arxiv.org/pdf/1807.00800.pdf>`_

    .. seealso:: :class:`~.LocalHilbertSchmidt`

    .. details::
        :title: Usage Details

        Consider that we want to evaluate the Hilbert-Schmidt Test cost between the unitary ``U`` and an approximate
        unitary ``V``. We need to define some functions where it is possible to use the :class:`~.HilbertSchmidt`
        template. Here the considered unitary is ``Hadamard`` and we try to compute the cost for the approximate
        unitary ``RZ``. For an angle that is equal to ``0`` (``Identity``), we have the maximal cost which is ``1``.

        .. code-block:: python

            u = qml.Hadamard(0)

            def v_function(params):
                qml.RZ(params[0], wires=1)

            dev = qml.device("default.qubit", wires=2)

            @qml.qnode(dev)
            def hilbert_test(v_params, v_function, v_wires, u):
                qml.HilbertSchmidt(v_params, v_function=v_function, v_wires=v_wires, u=u)
                return qml.probs(u.wires + v_wires)

            def cost_hst(parameters, v_function, v_wires, u):
                return (1 - hilbert_test(v_params=parameters, v_function=v_function, v_wires=v_wires, u=u)[0])

        Now that the cost function has been defined it can be called for specific parameters:

        >>> cost_hst([0], v_function = v_function, v_wires = [1], u = u)
        np.float64(1.0)

    """

    grad_method = None

    def _flatten(self):
        metadata = (
            ("v_function", self.hyperparameters["v_function"]),
            ("v_wires", self.hyperparameters["v_wires"]),
            ("u", self.hyperparameters["u"]),
        )
        return self.data, metadata

    @classmethod
    def _primitive_bind_call(cls, *params, v_function, v_wires, u, id=None):
        # pylint: disable=arguments-differ
        kwargs = {"v_function": v_function, "v_wires": v_wires, "u": u, "id": id}
        return cls._primitive.bind(*params, **kwargs)

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(*data, **dict(metadata))

    def __init__(
        self,
        V: Operation | Iterable[Operation],
        U: Operation | Iterable[Operation],
        id: str | None = None,
    ) -> None:

        u_ops = (U,) if isinstance(U, Operation) else tuple(U)
        if not all(isinstance(op, qml.operation.Operator) for op in u_ops):
            raise QuantumFunctionError(
                "The argument 'U' must be an Operator or an iterable of Operators."
            )
        self.hyperparameters["U"] = u_ops
        u_wires = qml.wires.Wires.all_wires([op.wires for op in u_ops])

        v_ops = (V,) if isinstance(V, Operation) else tuple(V)
        if not all(isinstance(op, qml.operation.Operator) for op in v_ops):
            raise QuantumFunctionError(
                "The argument 'V' must be an Operator or an iterable of Operators."
            )

        self.hyperparameters["V"] = v_ops
        v_wires = qml.wires.Wires.all_wires([op.wires for op in v_ops])

        if len(u_wires) != len(v_wires):
            raise QuantumFunctionError("U and V must have the same number of wires.")

        if len(qml.wires.Wires.shared_wires([u_wires, v_wires])) != 0:
            raise QuantumFunctionError("operations in u and v must act on distinct wires.")

        with qml.QueuingManager.stop_recording():
            params = qml.prod(*v_ops).parameters

        self._num_params = len(params)
        total_wires = qml.wires.Wires(u_wires + v_wires)

        super().__init__(*params, wires=total_wires, id=id)

    def map_wires(self, wire_map: dict):
        raise NotImplementedError("Mapping the wires of HilbertSchmidt is not implemented.")

    @property
    def num_params(self):
        return self._num_params

    @staticmethod
    def compute_decomposition(
        params: TensorLike,
        wires: int | Iterable[int | str] | qml.wires.Wires,
        U: Operation | Iterable[Operation],
        V: Operation | Iterable[Operation],
    ) -> list[Operation]:
        # pylint: disable=arguments-differ,unused-argument,too-many-positional-arguments
        r"""Representation of the operator as a product of other operators."""

        u_ops = (U,) if isinstance(U, Operation) else tuple(U)
        v_ops = (V,) if isinstance(V, Operation) else tuple(V)
        u_wires = qml.wires.Wires.all_wires([op.wires for op in u_ops])
        v_wires = qml.wires.Wires.all_wires([op.wires for op in v_ops])

        n_wires = len(u_wires + v_wires)
        first_range = range(n_wires // 2)
        second_range = range(n_wires // 2, n_wires)

        # Hadamard first layer
        decomp_ops = [qml.Hadamard(wires[i]) for i in first_range]
        # CNOT first layer
        decomp_ops.extend(
            qml.CNOT(wires=[wires[i], wires[j]]) for i, j in zip(first_range, second_range)
        )

        # Unitary U
        for op_u in u_ops:
            # The operation has been defined outside of this function, to queue it we call qml.apply.
            if qml.QueuingManager.recording():
                qml.apply(op_u)
            decomp_ops.append(op_u)

        # Unitary V conjugate
        # Since we don't currently have an easy way to apply the complex conjugate of a tape, we manually
        # apply the complex conjugate of each operation in the V tape and append it to the decomposition
        # using the QubitUnitary operation.
        for op_v in v_ops:
            mat = op_v.matrix().conjugate()
            decomp_ops.append(qml.QubitUnitary(mat, wires=op_v.wires))

        # CNOT second layer
        decomp_ops.extend(
            qml.CNOT(wires=[wires[i], wires[j]])
            for i, j in zip(reversed(first_range), reversed(second_range))
        )
        # Hadamard second layer
        decomp_ops.extend(qml.Hadamard(wires[i]) for i in first_range)
        return decomp_ops


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
        params (array): Parameters for the quantum function `V`.
        v_function (Callable): Quantum function that represents the approximate compiled unitary `V`.
        v_wires (int or Iterable[Number, str]]): the wire(s) on which the approximate compiled unitary acts.
        u (Operation or Iterable[Operation]): The operations that represent the unitary `U`.

    Raises:
        QuantumFunctionError: The argument ``u`` is not an Operator or an iterable of Operators.
        QuantumFunctionError: ``v_function`` is not a valid Quantum function.
        QuantumFunctionError: ``U`` and ``V`` do not have the same number of wires.
        QuantumFunctionError: The wires ``v_wires`` are a subset of `V` wires.
        QuantumFunctionError: Operations in ``u`` must act on distinct wires from those in ``v_wires``.

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

            u = qml.CZ(wires=(0,1))

            def v_function(params):
                qml.RZ(params[0], wires=2)
                qml.RZ(params[1], wires=3)
                qml.CNOT(wires=[2, 3])
                qml.RZ(params[2], wires=3)
                qml.CNOT(wires=[2, 3])

            dev = qml.device("default.qubit", wires=4)

            @qml.qnode(dev)
            def local_hilbert_test(v_params, v_function, v_wires, u):
                qml.LocalHilbertSchmidt(v_params, v_function=v_function, v_wires=v_wires, u=u)
                return qml.probs(u.wires + v_wires)

            def cost_lhst(parameters, v_function, v_wires, u):
                return (1 - local_hilbert_test(v_params=parameters, v_function=v_function, v_wires=v_wires, u=u)[0])

        Now that the cost function has been defined it can be called for specific parameters:

        >>> cost_lhst([3*np.pi/2, 3*np.pi/2, np.pi/2], v_function = v_function, v_wires = [2,3], u = u)
        np.float64(0.5)
    """

    @staticmethod
    def compute_decomposition(
        params: TensorLike,
        wires: int | Iterable[int | str] | qml.wires.Wires,
        u: Operation | Iterable[Operation],
        v: Operation | Iterable[Operation],
        v_function: Callable = None,
        v_wires: int | Iterable[int | str] | qml.wires.Wires = None,
    ) -> list[Operation]:
        # pylint: disable=too-many-positional-arguments
        r"""Representation of the operator as a product of other operators (static method)."""

        u_ops = (u,) if isinstance(u, Operation) else tuple(u)
        v_ops = (v,) if isinstance(v, Operation) else tuple(v)
        u_wires = qml.wires.Wires.all_wires([op.wires for op in u_ops])
        v_wires = qml.wires.Wires.all_wires([op.wires for op in v_ops])

        n_wires = len(u_wires + v_wires)
        first_range = range(n_wires // 2)
        second_range = range(n_wires // 2, n_wires)

        # Hadamard first layer
        decomp_ops = [qml.Hadamard(wires[i]) for i in first_range]
        # CNOT first layer
        decomp_ops.extend(
            qml.CNOT(wires=[wires[i], wires[j]]) for i, j in zip(first_range, second_range)
        )

        # Unitary U
        if qml.QueuingManager.recording():
            decomp_ops.extend(qml.apply(op_u) for op_u in u_ops)
        else:
            decomp_ops.extend(u_ops)

        # Unitary V conjugate
        # Since we don't currently have an easy way to apply the complex conjugate of a tape, we manually
        # apply the complex conjugate of each operation in the V tape and append it to the decomposition
        # using the QubitUnitary operation.
        decomp_ops.extend(
            qml.QubitUnitary(op_v.matrix().conjugate(), wires=op_v.wires) for op_v in v_ops
        )
        # Single qubit measurement
        decomp_ops.extend((qml.CNOT(wires=[wires[0], wires[n_wires // 2]]), qml.Hadamard(wires[0])))

        return decomp_ops
