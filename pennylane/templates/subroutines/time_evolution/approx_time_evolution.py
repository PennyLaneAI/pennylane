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
Contains the ApproxTimeEvolution template.
"""
# pylint: disable=protected-access
import copy
from collections import defaultdict

from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_resources, resource_rep
from pennylane.operation import Operation
from pennylane.ops import PauliRot
from pennylane.ops.op_math.linear_combination import Hamiltonian
from pennylane.pauli import PauliWord
from pennylane.queuing import QueuingManager, apply
from pennylane.wires import Wires, WiresLike


class ApproxTimeEvolution(Operation):
    r"""Applies the Trotterized time-evolution operator for an arbitrary Hamiltonian, expressed in terms
    of Pauli gates.

    .. note::

        We recommend using :class:`~.TrotterProduct` as the more general operation for approximate
        matrix exponentiation. One can recover the behaviour of :class:`~.ApproxTimeEvolution` by
        taking the adjoint:

        >>> qp.adjoint(qp.TrotterProduct(hamiltonian, time, order=1, n=n)) # doctest: +SKIP

    The general time-evolution operator for a time-independent Hamiltonian is given by

    .. math:: U(t) \ = \ e^{-i H t},

    for some Hamiltonian of the form:

    .. math:: H \ = \ \displaystyle\sum_{j} H_j.

    Implementing this unitary with a set of quantum gates is difficult, as the terms :math:`H_j` don't
    necessarily commute with one another. However, we are able to exploit the Trotter-Suzuki decomposition formula,

    .. math:: e^{A \ + \ B} \ = \ \lim_{n \to \infty} \Big[ e^{A/n} e^{B/n} \Big]^n,

    to implement an approximation of the time-evolution operator as

    .. math:: U \ \approx \ \displaystyle\prod_{k \ = \ 1}^{n} \displaystyle\prod_{j} e^{-i H_j t / n},

    with the approximation becoming better for larger :math:`n`.
    The circuit implementing this unitary is of the form:

    .. figure:: ../../_static/templates/subroutines/approx_time_evolution.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    It is also important to note that
    this decomposition is exact for any value of :math:`n` when each term of the Hamiltonian
    commutes with every other term.

    .. warning::

        The Trotter-Suzuki decomposition depends on the order of the summed observables. Two mathematically identical :class:`~.LinearCombination` objects may undergo different time evolutions
        due to the order in which those observables are stored.

    .. note::

       This template uses the :class:`~.PauliRot` operation in order to implement
       exponentiated terms of the input Hamiltonian. This operation only takes
       terms that are explicitly written in terms of products of Pauli matrices (:class:`~.PauliX`,
       :class:`~.PauliY`, :class:`~.PauliZ`, and :class:`~.Identity`).
       Thus, each term in the Hamiltonian must be expressed this way upon input, or else an error will be raised.

    Args:
        hamiltonian (.Hamiltonian): The Hamiltonian defining the
           time-evolution operator.
           The Hamiltonian must be explicitly written
           in terms of products of Pauli gates (:class:`~.PauliX`, :class:`~.PauliY`,
           :class:`~.PauliZ`, and :class:`~.Identity`).
        time (int or float): The time of evolution, namely the parameter :math:`t` in :math:`e^{- i H t}`.
        n (int): The number of Trotter steps used when approximating the time-evolution operator.

    .. seealso:: :class:`~.TrotterProduct`.

    .. details::
        :title: Usage Details

        The template is used inside a qnode:

        .. code-block:: python

            import pennylane as qp
            from pennylane import ApproxTimeEvolution

            n_wires = 2
            wires = range(n_wires)

            dev = qp.device('default.qubit', wires=n_wires)

            coeffs = [1, 1]
            obs = [qp.X(0), qp.X(1)]
            hamiltonian = qp.Hamiltonian(coeffs, obs)

            @qp.qnode(dev)
            def circuit(time):
                ApproxTimeEvolution(hamiltonian, time, 1)
                return [qp.expval(qp.Z(i)) for i in wires]

        >>> circuit(1)
        [np.float64(-0.416...), np.float64(-0.416...)]
    """

    grad_method = None

    resource_keys = {"words", "n"}

    def _flatten(self):
        h = self.hyperparameters["hamiltonian"]
        data = (h, self.data[-1])
        return data, (self.hyperparameters["n"],)

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(data[0], data[1], n=metadata[0])

    @property
    def resource_params(self) -> dict:
        return {
            "words": tuple(self.hyperparameters["hamiltonian"].pauli_rep.keys()),
            "n": self.hyperparameters["n"],
        }

    def __init__(self, hamiltonian, time, n, id=None):
        if getattr(hamiltonian, "pauli_rep", None) is None:
            raise ValueError(
                f"hamiltonian must be a linear combination of pauli words, got {type(hamiltonian).__name__}"
            )

        # extract the wires that the op acts on
        wires = hamiltonian.wires

        self._hyperparameters = {"hamiltonian": hamiltonian, "n": n}

        # trainable parameters are passed to the base init method
        super().__init__(*hamiltonian.data, time, wires=wires, id=id)

    def map_wires(self, wire_map: dict):
        new_op = copy.deepcopy(self)
        new_op._wires = Wires([wire_map.get(wire, wire) for wire in self.wires])
        new_op._hyperparameters["hamiltonian"] = new_op._hyperparameters["hamiltonian"].map_wires(
            wire_map
        )
        return new_op

    def queue(self, context=QueuingManager):
        context.remove(self.hyperparameters["hamiltonian"])
        context.append(self)
        return self

    @staticmethod
    def compute_decomposition(
        *coeffs_and_time, wires, hamiltonian, n
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.ApproxTimeEvolution.decomposition`.

        Args:
            *coeffs_and_time (TensorLike): coefficients of the Hamiltonian, appended by the time.
            wires (Any or Iterable[Any]): wires that the operator acts on
            hamiltonian (.Hamiltonian): The Hamiltonian defining the
               time-evolution operator. The Hamiltonian must be explicitly written
               in terms of products of Pauli gates (:class:`~.PauliX`, :class:`~.PauliY`,
               :class:`~.PauliZ`, and :class:`~.Identity`).
            n (int): The number of Trotter steps used when approximating the time-evolution operator.

        Returns:
            list[.Operator]: decomposition of the operator


        .. code-block:: python

            import pennylane as qp
            from pennylane import ApproxTimeEvolution

            num_qubits = 2

            hamiltonian = qp.Hamiltonian(
                [0.1, 0.2, 0.3], [qp.Z(0) @ qp.Z(1), qp.X(0), qp.X(1)]
            )

            evolution_time = 0.5
            trotter_steps = 1

            coeffs_and_time = [*hamiltonian.coeffs, evolution_time]


        >>> ApproxTimeEvolution.compute_decomposition(
        ...     *coeffs_and_time, wires=range(num_qubits), n=trotter_steps, hamiltonian=hamiltonian
        ... )
        [PauliRot(0.1, ZZ, wires=[0, 1]), PauliRot(0.2, X, wires=[0]), PauliRot(0.3, X, wires=[1])]
        """
        time = coeffs_and_time[-1]

        single_round = []
        with QueuingManager.stop_recording():
            for pw, coeff in hamiltonian.pauli_rep.items():
                if len(pw):
                    theta = 2 * time * coeff / n
                    term_str = "".join(pw.values())
                    wires = Wires(pw.keys())
                    single_round.append(PauliRot(theta, term_str, wires=wires))

        full_decomp = single_round * n
        if QueuingManager.recording():
            _ = [apply(op) for op in full_decomp]

        return full_decomp


def _approx_time_evolution_resources(words: tuple[PauliWord], n: int):
    resources = defaultdict(int)

    for _ in range(n):
        for pw in words:
            if len(pw) != 0:
                term_str = "".join(pw.values())
                resources[resource_rep(PauliRot, pauli_word=term_str)] += 1

    return resources


@register_resources(_approx_time_evolution_resources)
def _approx_time_evolution_decomposition(
    *coeffs_and_time: list, wires: WiresLike, hamiltonian: Hamiltonian, n: int
):  # pylint: disable=unused-argument
    time = coeffs_and_time[-1]

    @for_loop(n)
    def rounds_loop(_):

        for pauli_key in list(hamiltonian.pauli_rep.keys()):
            for pk, coeff in hamiltonian.pauli_rep.items():
                if pauli_key == pk:
                    break

            if len(pauli_key) != 0:
                theta = 2 * time * coeff / n  # pylint: disable=undefined-loop-variable
                term_str = "".join(pauli_key.values())
                wires = Wires(pauli_key.keys())
                PauliRot(theta, term_str, wires=wires)

    rounds_loop()  # pylint: disable=no-value-for-parameter


add_decomps(ApproxTimeEvolution, _approx_time_evolution_decomposition)
