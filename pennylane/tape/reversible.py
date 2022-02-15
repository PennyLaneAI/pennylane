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
"""
Quantum tape that implements reversible backpropagation.
"""
# pylint: disable=attribute-defined-outside-init,protected-access
import copy
from functools import reduce
from string import ascii_letters as ABC
import warnings

import numpy as np

import pennylane as qml

from .jacobian_tape import JacobianTape
from .tape import QuantumTape


ABC_ARRAY = np.array(list(ABC))


class ReversibleTape(JacobianTape):
    r"""Quantum tape for computing gradients via reversible analytic differentiation.

    .. note::

        The reversible analytic differentiation method has the following restrictions:

        * As it requires knowledge of the statevector, only statevector simulator devices can be used.

        * Differentiation is only supported for the parametrized quantum operations
          :class:`~.RX`, :class:`~.RY`, :class:`~.RZ`, and :class:`~.Rot`.

    This class extends the :class:`~.jacobian` method of the quantum tape to support analytic
    gradients of qubit operations using reversible analytic differentiation. This gradient method
    returns *exact* gradients, however requires use of a statevector simulator. Simply create
    the tape, and then call the Jacobian method:

    >>> tape.jacobian(dev)

    For more details on the quantum tape, please see :class:`~.JacobianTape`.

    **Reversible analytic differentiation**

    Assume a circuit has a gate :math:`G(\theta)` that we want to differentiate.
    Without loss of generality, we can write the circuit in the form three unitaries: :math:`UGV`.
    Starting from the initial state :math:`\vert 0\rangle`, the quantum state is evolved up to the
    "pre-measurement" state :math:`\vert\psi\rangle=UGV\vert 0\rangle`, which is saved
    (this can be reused for each variable being differentiated).

    We then apply the unitary :math:`V^{-1}` to evolve this state backwards in time
    until just after the gate :math:`G` (hence the name "reversible").
    The generator of :math:`G` is then applied as a gate, and we evolve forward using :math:`V` again.
    At this stage, the state of the simulator is proportional to
    :math:`\frac{\partial}{\partial\theta}\vert\psi\rangle`.
    Some further post-processing of this gives the derivative
    :math:`\frac{\partial}{\partial\theta} \langle \hat{O} \rangle` for any observable O.

    The reversible approach is similar to backpropagation, but trades off extra computation for
    enhanced memory efficiency. Where backpropagation caches the state tensors at each step during
    a forward pass, the reversible method only caches the final pre-measurement state.

    Compared to the parameter-shift rule, the reversible method can
    be faster or slower, depending on the density and location of parametrized gates in a circuit
    (circuits with higher density of parametrized gates near the end of the circuit will see a
    benefit).
    """

    def _grad_method(self, idx, use_graph=True, default_method="A"):
        return super()._grad_method(idx, use_graph=use_graph, default_method=default_method)

    @staticmethod
    def _matrix_elem(vec1, obs, vec2, dev_wires):
        r"""Computes the matrix element of an observable.

        That is, given two basis states :math:`\mathbf{i}`, :math:`\mathbf{j}`,
        this method returns :math:`\langle \mathbf{i} \vert \hat{O} \vert \mathbf{j} \rangle`.
        Unmeasured wires are contracted, and a scalar is returned.

        Args:
            vec1 (array[complex]): a length :math:`2^N` statevector
            obs (.Observable): a PennyLane observable
            vec2 (array[complex]): a length :math:`2^N` statevector
            dev_wires (pennylane.wires.Wires): wires of the device used to prepare the state
        """
        # pylint: disable=protected-access

        mat = np.reshape(obs.get_matrix(), [2] * len(obs.wires) * 2)
        vec1 = np.reshape(vec1, [2] * len(dev_wires))
        vec2 = np.reshape(vec2, [2] * len(dev_wires))

        vec1_indices = ABC[: len(dev_wires)]

        # compute the indices of the observable's wires on the device
        wire_indices = dev_wires.indices(obs.wires)
        obs_in_indices = "".join(ABC_ARRAY[wire_indices].tolist())
        obs_out_indices = ABC[len(dev_wires) : len(dev_wires) + len(obs.wires)]
        obs_indices = "".join([obs_in_indices, obs_out_indices])

        vec2_indices = reduce(
            lambda old_string, idx_pair: old_string.replace(idx_pair[0], idx_pair[1]),
            zip(obs_in_indices, obs_out_indices),
            vec1_indices,
        )

        einsum_str = f"{vec1_indices},{obs_indices},{vec2_indices}->"

        return np.einsum(einsum_str, np.conj(vec1), mat, vec2)

    def reversible_diff(self, idx, params, **options):
        """Generate the tapes and postprocessing methods required to compute the gradient of a
        parameter using the reversible backpropagation method.

        Args:
            idx (int): trainable parameter index to differentiate with respect to
            params (list[Any]): the quantum tape operation parameters

        Keyword Args:
            dev_wires (.Wires): wires on the device the reversible backpropagation method
                is computed on

        Returns:
            tuple[list[QuantumTape], function]: A tuple containing the list of generated tapes,
            in addition to a post-processing function to be applied to the evaluated
            tapes.
        """

        # The reversible tape only support differentiating
        # expectation values of observables for now.
        for m in self.measurements:
            if (
                m.return_type is qml.operation.Variance
                or m.return_type is qml.operation.Probability
            ):
                raise ValueError(
                    f"{m.return_type} is not supported with the reversible gradient method"
                )
            if m.obs.name == "Hamiltonian":
                raise qml.QuantumFunctionError(
                    "Reverse differentiation method does not support Hamiltonian observables."
                )

        t_idx = list(self.trainable_params)[idx]
        op = self._par_info[t_idx]["op"]
        p_idx = self._par_info[t_idx]["p_idx"]

        # The reversible tape only supports the RX, RY, RZ, and Rot operations for now:
        #
        # * CRX, CRY, CRZ ops have a non-unitary matrix as generator.
        #
        # * PauliRot, MultiRZ, U2, and U3 do not have generators specified.
        #
        # TODO: the controlled rotations can be supported by multiplying ``state``
        # directly by these generators within this function
        # (or by allowing non-unitary matrix multiplies in the simulator backends)

        if op.name not in ["RX", "RY", "RZ", "Rot"]:
            raise ValueError(
                f"The {op.name} gate is not currently supported with the "
                f"reversible gradient method."
            )

        # get the stored final state of the original circuit, which we start from here

        final_state = self._final_state
        # get the wires on the device used for the differentiation

        dev_wires = options.get("dev_wires")

        self.set_parameters(params)

        # create a new circuit which rewinds the pre-measurement state to just after `op`,
        # applies the generator of `op`, and then plays forward back to
        # pre-measurement step
        op_idx = self.operations.index(op)
        between_ops = self.operations[op_idx + 1 :]

        if op.name == "Rot":
            decomp = op.decomposition()
            generator, multiplier = qml.utils.get_generator(decomp[p_idx])
            between_ops = decomp[p_idx + 1 :] + between_ops
        else:
            generator, multiplier = qml.utils.get_generator(op)

        # construct circuit to compute differentiated state
        between_ops_inverse = [copy.copy(op) for op in between_ops[::-1]]

        with QuantumTape() as new_circuit:
            # start with final state of original circuit
            qml.QubitStateVector(final_state, wires=dev_wires)

            # evolve circuit backwards until gate we want to differentiate
            for op in between_ops_inverse:
                op.queue().inv()

            # apply generator needed for differentiation
            qml.apply(generator)

            # evolve forwards again
            for op in between_ops:
                op.queue()

            qml.state()

        tapes = [new_circuit]

        def processing_fn(results):
            """Computes the gradient of the parameter at index idx via the
            reversible backprop method.

            Args:
                results (list[real]): evaluated quantum tapes

            Returns:
                array[float]: 1-dimensional array of length determined by the tape output
                measurement statistics
            """
            dstate = results[0][0]

            # compute matrix element <d(state)|O|state> for each observable O
            # TODO: if all observables act on same number of wires, could do all at once with einsum
            matrix_elems = [
                self._matrix_elem(dstate, ob, final_state, dev_wires) for ob in self.observables
            ]
            matrix_elems = np.array(matrix_elems)
            return 2 * multiplier * np.imag(matrix_elems)

        return tapes, processing_fn

    def jacobian(self, device, params=None, **options):
        # The reversible_diff method needs to evaluate the circuit
        # at the unshifted parameter values; the pre-rotated statevector is then stored
        # in the self._state attribute. Here, we set the value of the attribute to None

        # before each Jacobian call, so that the statevector is calculated only once.
        self._final_state = None
        if device.shots is not None:
            warnings.warn(
                "Requested reversible differentiation to be computed with finite shots."
                " Reversible differentiation always calculated exactly.",
                UserWarning,
            )

        return super().jacobian(device, params, **options)

    def analytic_pd(self, idx, params, **options):
        device = options["device"]

        # circuits constructed in reversible differentiation always start
        # with the final state of the original circuit, which we store here
        if self._final_state is None:
            self.execute_device(params, device)
            # todo: better create a new tape that has state as output method here?
            self._final_state = device._pre_rotated_state.flatten()

        # we need the wires to prepare the final state in each run, and to
        # be able to compute an expecation value by hand
        options["dev_wires"] = device.wires

        return self.reversible_diff(idx, params=params, **options)
