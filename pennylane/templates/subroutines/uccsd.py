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
Contains the UCCSD template.
"""
# pylint: disable-msg=too-many-arguments,protected-access,too-many-positional-arguments
import copy
from collections import Counter

import numpy as np

from pennylane import capture, math
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_resources, resource_rep
from pennylane.operation import Operation
from pennylane.ops import BasisState
from pennylane.wires import Wires

from .fermionic_double_excitation import FermionicDoubleExcitation
from .fermionic_single_excitation import FermionicSingleExcitation

has_jax = True
try:
    from jax import numpy as jnp
except (ModuleNotFoundError, ImportError) as import_error:  # pragma: no cover
    has_jax = False  # pragma: no cover


class UCCSD(Operation):
    r"""Implements the Unitary Coupled-Cluster Singles and Doubles (UCCSD) ansatz.

    The UCCSD ansatz calls the
    :func:`~.FermionicSingleExcitation` and :func:`~.FermionicDoubleExcitation`
    templates to exponentiate the coupled-cluster excitation operator. UCCSD is a VQE ansatz
    commonly used to run quantum chemistry simulations.

    The UCCSD unitary, within the first-order Trotter approximation, is given by:

    .. math::

        \hat{U}(\vec{\theta}) =
        \prod_{p > r} \mathrm{exp} \Big\{\theta_{pr}
        (\hat{c}_p^\dagger \hat{c}_r-\mathrm{H.c.}) \Big\}
        \prod_{p > q > r > s} \mathrm{exp} \Big\{\theta_{pqrs}
        (\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s-\mathrm{H.c.}) \Big\}

    where :math:`\hat{c}` and :math:`\hat{c}^\dagger` are the fermionic annihilation and
    creation operators and the indices :math:`r, s` and :math:`p, q` run over the occupied and
    unoccupied molecular orbitals, respectively. Using the `Jordan-Wigner transformation
    <https://arxiv.org/abs/1208.5986>`_ the UCCSD unitary defined above can be written in terms
    of Pauli matrices as follows (for more details see
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_):

    .. math::

        \hat{U}(\vec{\theta}) = && \prod_{p > r} \mathrm{exp} \Big\{ \frac{i\theta_{pr}}{2}
        \bigotimes_{a=r+1}^{p-1} \hat{Z}_a (\hat{Y}_r \hat{X}_p - \mathrm{H.c.}) \Big\} \\
        && \times \prod_{p > q > r > s} \mathrm{exp} \Big\{ \frac{i\theta_{pqrs}}{8}
        \bigotimes_{b=s+1}^{r-1} \hat{Z}_b \bigotimes_{a=q+1}^{p-1}
        \hat{Z}_a (\hat{X}_s \hat{X}_r \hat{Y}_q \hat{X}_p +
        \hat{Y}_s \hat{X}_r \hat{Y}_q \hat{Y}_p +
        \hat{X}_s \hat{Y}_r \hat{Y}_q \hat{Y}_p +
        \hat{X}_s \hat{X}_r \hat{X}_q \hat{Y}_p -
        \{\mathrm{H.c.}\}) \Big\}.

    Args:
        weights (tensor_like): Size ``(n_repeats, len(s_wires) + len(d_wires),)`` tensor containing the
            parameters (see usage details below) :math:`\theta_{pr}` and :math:`\theta_{pqrs}` entering
            the Z rotation in :func:`~.FermionicSingleExcitation` and :func:`~.FermionicDoubleExcitation`.
            These parameters are the coupled-cluster amplitudes that need to be optimized for each
            single and double excitation generated with the :func:`~.excitations` function.
            If the size of the given tensor is ``(len(s_wires) + len(d_wires),)``, it is assumed that ``n_repeats == 1``.
            :math:`\theta_{pr}` and :math:`\theta_{pqrs}` entering the Z rotation in
            :func:`~.FermionicSingleExcitation` and :func:`~.FermionicDoubleExcitation`.
            These parameters are the coupled-cluster amplitudes that need to be optimized for each
            single and double excitation generated with the :func:`~.excitations` function.
        wires (Iterable): wires that the template acts on
        s_wires (Sequence[Sequence]): Sequence of lists containing the wires ``[r,...,p]``
            resulting from the single excitation
            :math:`\vert r, p \rangle = \hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF} \rangle`,
            where :math:`\vert \mathrm{HF} \rangle` denotes the Hartee-Fock reference state.
            The first (last) entry ``r`` (``p``) is considered the wire representing the
            occupied (unoccupied) orbital where the electron is annihilated (created).
        d_wires (Sequence[Sequence[Sequence]]): Sequence of lists, each containing two lists that
            specify the indices ``[s, ...,r]`` and ``[q,..., p]`` defining the double excitation
            :math:`\vert s, r, q, p \rangle = \hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r
            \hat{c}_s \vert \mathrm{HF} \rangle`. The entries ``s`` and ``r`` are wires
            representing two occupied orbitals where the two electrons are annihilated
            while the entries ``q`` and ``p`` correspond to the wires representing two unoccupied
            orbitals where the electrons are created. Wires in-between represent the occupied
            and unoccupied orbitals in the intervals ``[s, r]`` and ``[q, p]``, respectively.
        init_state (array[int]): Length ``len(wires)`` occupation-number vector representing the
            HF state. ``init_state`` is used to initialize the wires.
        n_repeats (int): Number of times the UCCSD unitary is repeated. The default value is ``1``.

    .. details::
        :title: Usage Details

        Notice that:

        #. The number of wires has to be equal to the number of spin orbitals included in
           the active space.

        #. The single and double excitations can be generated with the function
           :func:`~.excitations`. See example below.

        #. The vector of parameters ``weights`` is a two-dimensional array of shape
           ``(n_repeats, len(s_wires)+len(d_wires))``.
        #. If ``n_repeats=1``, then ``weights`` can also be a one-dimensional array of shape
           ``(len(s_wires)+len(d_wires),)``.


        An example of how to use this template is shown below:

        .. code-block:: python

            import pennylane as qml
            from pennylane import numpy as np

            # Define the molecule
            symbols  = ['H', 'H', 'H']
            geometry = np.array([[0.01076341,  0.04449877,  0.0],
                                 [0.98729513,  1.63059094,  0.0],
                                 [1.87262415, -0.00815842,  0.0]], requires_grad = False)
            electrons = 2
            charge = 1

            # Build the electronic Hamiltonian
            H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, charge=charge)

            # Define the HF state
            hf_state = qml.qchem.hf_state(electrons, qubits)

            # Generate single and double excitations
            singles, doubles = qml.qchem.excitations(electrons, qubits)

            # Map excitations to the wires the UCCSD circuit will act on
            s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

            # Define the device
            dev = qml.device("default.qubit", wires=qubits)

            # Define the qnode
            @qml.qnode(dev)
            def circuit(params, wires, s_wires, d_wires, hf_state):
                qml.UCCSD(params, wires, s_wires, d_wires, hf_state)
                return qml.expval(H)

            # Define the initial values of the circuit parameters
            params = np.zeros(len(singles) + len(doubles))

            # Define the optimizer
            optimizer = qml.GradientDescentOptimizer(stepsize=0.5)

            # Optimize the circuit parameters and compute the energy
            for n in range(21):
                params, energy = optimizer.step_and_cost(circuit, params,
                wires=range(qubits), s_wires=s_wires, d_wires=d_wires, hf_state=hf_state)
                if n % 2 == 0:
                    print("step = {:},  E = {:.8f} Ha".format(n, energy))

        .. code-block:: none

            step = 0,  E = -1.24654994 Ha
            step = 2,  E = -1.27016844 Ha
            step = 4,  E = -1.27379541 Ha
            step = 6,  E = -1.27434106 Ha
            step = 8,  E = -1.27442311 Ha
            step = 10,  E = -1.27443547 Ha
            step = 12,  E = -1.27443733 Ha
            step = 14,  E = -1.27443761 Ha
            step = 16,  E = -1.27443765 Ha
            step = 18,  E = -1.27443766 Ha
            step = 20,  E = -1.27443766 Ha

    """

    grad_method = None

    resource_keys = {"num_wires", "n_repeats", "num_d_wires", "num_s_wires"}

    def __init__(
        self, weights, wires, s_wires=None, d_wires=None, init_state=None, n_repeats=1, id=None
    ):
        if (not s_wires) and (not d_wires):
            raise ValueError(
                f"s_wires and d_wires lists can not be both empty; got ph={s_wires}, pphh={d_wires}"
            )

        for d_wires_ in d_wires:
            if len(d_wires_) != 2:
                raise ValueError(
                    f"expected entries of d_wires to be of size 2; got {d_wires_} of length {len(d_wires_)}"
                )

        if n_repeats < 1:
            raise ValueError(f"Requires n_repeats to be at least 1; got {n_repeats}.")

        shape = math.shape(weights)

        expected_shape = (len(s_wires) + len(d_wires),)
        if len(shape) == 1 and (n_repeats != 1 or shape != expected_shape):
            raise ValueError(
                f"For one-dimensional weights tensor, the shape must be {expected_shape}, and n_repeats should be 1; "
                f"got {shape} and {n_repeats}, respectively."
            )
        if len(shape) != 1 and shape != (n_repeats,) + expected_shape:
            raise ValueError(
                f"Weights tensor must be of shape {(n_repeats,) + expected_shape}; got {shape}."
            )

        init_state = math.toarray(init_state)

        if init_state.dtype != np.dtype("int"):
            raise ValueError(f"Elements of 'init_state' must be integers; got {init_state.dtype}")

        self._hyperparameters = {
            "init_state": tuple(init_state),
            "s_wires": tuple(tuple(w) for w in s_wires),
            "d_wires": tuple(tuple(tuple(w) for w in dw) for dw in d_wires),
            "n_repeats": n_repeats,
        }

        super().__init__(weights, wires=wires, id=id)

    def map_wires(self, wire_map: dict):
        new_op = copy.deepcopy(self)
        new_op._wires = Wires([wire_map.get(wire, wire) for wire in self.wires])
        new_op._hyperparameters["s_wires"] = tuple(
            tuple(wire_map.get(w, w) for w in wires) for wires in self._hyperparameters["s_wires"]
        )
        new_op._hyperparameters["d_wires"] = tuple(
            tuple(tuple(wire_map.get(w, w) for w in _wires) for _wires in wires)
            for wires in self._hyperparameters["d_wires"]
        )
        return new_op

    @property
    def resource_params(self) -> dict:
        return {
            "num_d_wires": [
                (len(x[0]), len(x[1])) if len(x) else () for x in self.hyperparameters["d_wires"]
            ],
            "num_s_wires": [len(x) for x in self.hyperparameters["s_wires"]],
            "n_repeats": self.hyperparameters["n_repeats"],
            "num_wires": len(self.wires),
        }

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(
        weights, wires, s_wires, d_wires, init_state, n_repeats
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.UCCSD.decomposition`.

        Args:
            weights (tensor_like): Size ``(len(s_wires) + len(d_wires),)`` or ``(n_repeats, len(s_wires) + len(d_wires),)``,
                depending on ``n_repeats``, tensor containing the parameters
                entering the Z rotation in :func:`~.FermionicSingleExcitation` and :func:`~.FermionicDoubleExcitation`.
            wires (Any or Iterable[Any]): wires that the operator acts on
            s_wires (Sequence[Sequence]): Sequence of lists containing the wires ``[r,...,p]``
                resulting from the single excitation.
            d_wires (Sequence[Sequence[Sequence]]): Sequence of lists, each containing two lists that
                specify the indices ``[s, ...,r]`` and ``[q,..., p]`` defining the double excitation.
            init_state (array[int]): Length ``len(wires)`` occupation-number vector representing the
                HF state. ``init_state`` is used to initialize the wires.
            n_repeats (int): Number of times the UCCSD unitary is repeated.

        Returns:
            list[.Operator]: decomposition of the operator
        """
        op_list = []

        op_list.append(BasisState(init_state, wires=wires))

        if n_repeats == 1 and len(math.shape(weights)) == 1:
            weights = math.expand_dims(weights, 0)

        for layer in range(n_repeats):
            for i, (w1, w2) in enumerate(d_wires):
                op_list.append(
                    FermionicDoubleExcitation(
                        weights[layer][len(s_wires) + i], wires1=w1, wires2=w2
                    )
                )

            for j, s_wires_ in enumerate(s_wires):
                op_list.append(FermionicSingleExcitation(weights[layer][j], wires=s_wires_))

        return op_list


def _UCCSD_resources(num_wires, n_repeats, num_d_wires, num_s_wires):
    resources = Counter(
        {
            resource_rep(BasisState, num_wires=num_wires): 1,
        }
    )

    for _ in range(n_repeats):
        for w1, w2 in num_d_wires:
            resources[resource_rep(FermionicDoubleExcitation, num_wires_1=w1, num_wires_2=w2)] += 1

        for s in num_s_wires:
            resources[resource_rep(FermionicSingleExcitation, num_wires=s)] += 1

    return dict(resources)


@register_resources(_UCCSD_resources)
def _UCCSD_decomposition(weights, wires, s_wires, d_wires, init_state, n_repeats):
    BasisState(init_state, wires=wires)

    if n_repeats == 1 and len(math.shape(weights)) == 1:
        weights = math.expand_dims(weights, 0)

    if has_jax and capture.enabled():
        weights, d_wires, s_wires = jnp.array(weights), jnp.array(d_wires), jnp.array(s_wires)

    @for_loop(n_repeats)
    def apply_layers(layer):
        @for_loop(len(d_wires))
        def double_excitation(i):
            (w1, w2) = d_wires[i]
            FermionicDoubleExcitation(weights[layer][len(s_wires) + i], wires1=w1, wires2=w2)

        @for_loop(len(s_wires))
        def single_excitation(j):
            s_wires_ = s_wires[j]
            FermionicSingleExcitation(weights[layer][j], wires=s_wires_)

        double_excitation()  # pylint: disable=no-value-for-parameter
        single_excitation()  # pylint: disable=no-value-for-parameter

    apply_layers()  # pylint: disable=no-value-for-parameter


add_decomps(UCCSD, _UCCSD_decomposition)
