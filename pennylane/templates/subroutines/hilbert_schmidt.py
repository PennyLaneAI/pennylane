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
# pylint: disable-msg=too-many-arguments
import pennylane as qml
from pennylane.operation import AnyWires, Operation


class HilbertSchmidt(Operation):
    r"""Create a Hilbert-Schmidt template that can be used to compute the Hilbert-Schmidt Test (HST).

    The HST is a useful quantity used when we want to compile an unitary `U` with an approximate unitary `V`. The HST
    is used as a distance between `U` and `V`, the result of executing the HST is 0 if and only if `V` is equal to
    `U` (up to a global phase). Therefore we can define a cost by:

    .. math::
        C_{HST} = 1 - \frac{1}{d^2} \left|Tr(V^{\dagger}U)\right|^2,

    where the quantity :math:`\frac{1}{d^2} \left|Tr(V^{\dagger}U)\right|^2` is obtained by executing the
    Hilbert-Schmidt Test. It is equivalent to taking the outcome probability of the state :math:`|0 ... 0\rangle`
    for the following circuit:

    .. figure:: ../../_static/templates/subroutines/hst.png
        :align: center
        :width: 80%
        :target: javascript:void(0);

    It defines our decomposition for the Hilbert-Schmidt Test template.

    Args:
        params (array): Parameters for the quantum function `V`.
        v_function (callable): Quantum function that represents the approximate compiled unitary `V`.
        v_wires (int or Iterable[Number, str]]): The wire(s) the approximate compiled unitary act on.
        u_tape (.QuantumTape): `U`, the unitary to be compiled as a ``qml.tape.QuantumTape``.

    Raises:
        QuantumFunctionError: The argument ``u_tape`` must be a ``QuantumTape``.
        QuantumFunctionError: ``v_function`` is not a valid quantum function.
        QuantumFunctionError: ``U`` and ``V`` do not have the same number of wires.
        QuantumFunctionError: The wires ``v_wires`` are a subset of ``V`` wires.
        QuantumFunctionError: ``u_tape`` and ``v_tape`` must act on distinct wires.

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

            with qml.tape.QuantumTape(do_queue=False) as u_tape:
                qml.Hadamard(wires=0)

            def v_function(params):
                qml.RZ(params[0], wires=1)

            dev = qml.device("default.qubit", wires=2)

            @qml.qnode(dev)
            def hilbert_test(v_params, v_function, v_wires, u_tape):
                qml.HilbertSchmidt(v_params, v_function=v_function, v_wires=v_wires, u_tape=u_tape)
                return qml.probs(u_tape.wires + v_wires)

            def cost_hst(parameters, v_function, v_wires, u_tape):
                return (1 - hilbert_test(v_params=parameters, v_function=v_function, v_wires=v_wires, u_tape=u_tape)[0])

        Now that the cost function has been defined it can be called for specific parameters:

        >>> cost_hst([0], v_function = v_function, v_wires = [1], u_tape = u_tape)
        1

    """
    num_wires = AnyWires
    grad_method = None

    def __init__(self, *params, v_function, v_wires, u_tape, do_queue=True, id=None):

        self._num_params = len(params)

        if not isinstance(u_tape, qml.tape.QuantumTape):
            raise qml.QuantumFunctionError("The argument u_tape must be a QuantumTape.")

        u_wires = u_tape.wires
        self.hyperparameters["u_tape"] = u_tape

        if not callable(v_function):
            raise qml.QuantumFunctionError(
                "The argument v_function must be a callable quantum function."
            )

        self.hyperparameters["v_function"] = v_function

        v_tape = qml.transforms.make_tape(v_function)(*params)
        self.hyperparameters["v_tape"] = v_tape
        self.hyperparameters["v_wires"] = v_tape.wires

        if len(u_wires) != len(v_wires):
            raise qml.QuantumFunctionError("U and V must have the same number of wires.")

        if not qml.wires.Wires(v_wires).contains_wires(v_tape.wires):
            raise qml.QuantumFunctionError("All wires in v_tape must be in v_wires.")

        # Intersection of wires
        if len(qml.wires.Wires.shared_wires([u_tape.wires, v_tape.wires])) != 0:
            raise qml.QuantumFunctionError("u_tape and v_tape must act on distinct wires.")

        wires = qml.wires.Wires(u_wires + v_wires)

        super().__init__(*params, wires=wires, do_queue=do_queue, id=id)

    @property
    def num_params(self):
        return self._num_params

    @staticmethod
    def compute_decomposition(
        params, wires, u_tape, v_tape, v_function=None, v_wires=None
    ):  # pylint: disable=arguments-differ,unused-argument
        r"""Representation of the operator as a product of other operators."""
        n_wires = len(u_tape.wires + v_tape.wires)
        decomp_ops = []

        first_range = range(0, int(n_wires / 2))
        second_range = range(int(n_wires / 2), n_wires)

        # Hadamard first layer
        for i in first_range:
            decomp_ops.append(qml.Hadamard(wires[i]))

        # CNOT first layer
        for i, j in zip(first_range, second_range):
            decomp_ops.append(qml.CNOT(wires=[wires[i], wires[j]]))

        # Unitary U
        for op_u in u_tape.operations:
            # The operation has been defined outside of this function, to queue it we call qml.apply.
            qml.apply(op_u)
            decomp_ops.append(op_u)

        # Unitary V conjugate
        for op_v in v_tape.operations:
            decomp_ops.append(qml.adjoint(op_v, lazy=False))

        # CNOT second layer
        for i, j in zip(reversed(first_range), reversed(second_range)):
            decomp_ops.append(qml.CNOT(wires=[wires[i], wires[j]]))

        # Hadamard second layer
        for i in first_range:
            decomp_ops.append(qml.Hadamard(wires[i]))
        return decomp_ops


class LocalHilbertSchmidt(HilbertSchmidt):
    r"""Create a Local Hilbert-Schmidt template that can be used to compute the  Local Hilbert-Schmidt Test (LHST).
    The result of the LHST is a useful quantity for compiling a unitary ``U`` with an approximate unitary ``V``. The
    LHST is used as a distance between `U` and `V`, it is similar to the Hilbert-Schmidt test, but the measurement is
    made only on one qubit at the end of the circuit. The LHST cost is always smaller than the HST cost and is useful
    for large unitaries.

    .. figure:: ../../_static/templates/subroutines/lhst.png
        :align: center
        :width: 80%
        :target: javascript:void(0);

    Args:
        params (array): Parameters for the quantum function `V`.
        v_function (Callable): Quantum function that represents the approximate compiled unitary `V`.
        v_wires (int or Iterable[Number, str]]): the wire(s) the approximate compiled unitary act on.
        u_tape (.QuantumTape): `U`, the unitary to be compiled as a ``qml.tape.QuantumTape``.

    Raises:
        QuantumFunctionError: The argument u_tape must be a QuantumTape
        QuantumFunctionError: ``v_function`` is not a valid Quantum function.
        QuantumFunctionError: `U` and `V` do not have the same number of wires.
        QuantumFunctionError: The wires ``v_wires`` are a subset of `V` wires.
        QuantumFunctionError: u_tape and v_tape must act on distinct wires.

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

            with qml.tape.QuantumTape(do_queue=False) as u_tape:
                qml.CZ(wires=[0,1])

            def v_function(params):
                qml.RZ(params[0], wires=2)
                qml.RZ(params[1], wires=3)
                qml.CNOT(wires=[2, 3])
                qml.RZ(params[2], wires=3)
                qml.CNOT(wires=[2, 3])

            dev = qml.device("default.qubit", wires=4)

            @qml.qnode(dev)
            def local_hilbert_test(v_params, v_function, v_wires, u_tape):
                qml.LocalHilbertSchmidt(v_params, v_function=v_function, v_wires=v_wires, u_tape=u_tape)
                return qml.probs(u_tape.wires + v_wires)

            def cost_lhst(parameters, v_function, v_wires, u_tape):
                return (1 - local_hilbert_test(v_params=parameters, v_function=v_function, v_wires=v_wires, u_tape=u_tape)[0])

        Now that the cost function has been defined it can be called for specific parameters:

        >>> cost_lhst([3*np.pi/2, 3*np.pi/2, np.pi/2], v_function = v_function, v_wires = [1], u_tape = u_tape)
        0.5
    """

    @staticmethod
    def compute_decomposition(
        params, wires, u_tape, v_tape, v_function=None, v_wires=None
    ):  # pylint: disable=arguments-differ,unused-argument
        r"""Representation of the operator as a product of other operators (static method)."""
        decomp_ops = []
        n_wires = len(u_tape.wires + v_tape.wires)
        first_range = range(0, int(n_wires / 2))
        second_range = range(int(n_wires / 2), n_wires)

        # Hadamard first layer
        for i in first_range:
            decomp_ops.append(qml.Hadamard(wires[i]))

        # CNOT first layer
        for i, j in zip(first_range, second_range):
            decomp_ops.append(qml.CNOT(wires=[wires[i], wires[j]]))

        # Unitary U
        for op_u in u_tape.operations:
            qml.apply(op_u)
            decomp_ops.append(op_u)

        # Unitary V conjugate
        for op_v in v_tape.operations:
            decomp_ops.append(qml.adjoint(qml.apply, lazy=False)(op_v))

        # Only one CNOT
        decomp_ops.append(qml.CNOT(wires=[wires[0], wires[int(n_wires / 2)]]))

        # Only one Hadamard
        decomp_ops.append(qml.Hadamard(wires[0]))

        return decomp_ops
