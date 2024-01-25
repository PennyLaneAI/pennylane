# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
This module contains the Clifford simulator using ``stim``.
"""

from dataclasses import replace
from functools import partial
from numbers import Number
from typing import Union, Tuple, Sequence
import concurrent.futures
import numpy as np

import pennylane as qml
from pennylane.tape import QuantumTape, QuantumScript
from pennylane.typing import Result, ResultBatch
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.transforms.core import TransformProgram
from pennylane.measurements import (
    ExpectationMP,
    StateMP,
    DensityMatrixMP,
    PurityMP,
    VnEntropyMP,
    MutualInfoMP,
    VarianceMP,
    ProbabilityMP,
    SampleMP,
    ClassicalShadowMP,
    ShadowExpvalMP,
)
from pennylane.ops.op_math import Prod, SProd, Sum
from pennylane.devices.qubit.sampling import get_num_shots_and_executions

from . import Device
from .execution_config import ExecutionConfig, DefaultExecutionConfig

from .default_qubit import accepted_sample_measurement

from .preprocess import (
    decompose,
    validate_observables,
    validate_measurements,
    validate_multiprocessing_workers,
    validate_device_wires,
    validate_adjoint_trainable_params,
)

Result_or_ResultBatch = Union[Result, ResultBatch]
QuantumTape_or_Batch = Union[QuantumTape, Sequence[QuantumTape]]

# Updated observable list
_MEAS_OBSERVABLES = {
    "PauliX",
    "PauliY",
    "PauliZ",
    "Hadamard",
    "Hermitian",
    "Identity",
    "Projector",
    "Hamiltonian",
    "Sum",
    "SProd",
    "Prod",
}

# Clifford gates
_GATE_OPERATIONS = {
    "Identity": "I",
    "PauliX": "X",
    "PauliY": "Y",
    "PauliZ": "Z",
    "Hadamard": "H",
    "S": "S",
    "Adjoint(S)": "S_DAG",
    "SX": "SX",
    "Adjoint(SX)": "SX_DAG",
    "CNOT": "CNOT",
    "SWAP": "SWAP",
    "ISWAP": "ISWAP",
    "Adjoint(ISWAP)": "ISWAP_DAG",
    "CY": "CY",
    "CZ": "CZ",
    "GlobalPhase": None,
    "BasisState": None,
    "StatePrep": None,
    "Snapshot": None,
    "Barrier": None,
}


def operation_stopping_condition(op: qml.operation.Operator) -> bool:
    """Specifies whether an operation is accepted by ``DefaultClifford``."""
    return op.name in _GATE_OPERATIONS


def observable_stopping_condition(obs: qml.operation.Operator) -> bool:
    """Specifies whether an observable is accepted by ``DefaultClifford``."""
    return obs.name in _MEAS_OBSERVABLES


def _import_stim():
    """Import stim."""
    try:
        # pylint: disable=import-outside-toplevel, unused-import, multiple-imports
        import stim
    except ImportError as Error:
        raise ImportError(
            "This feature requires stim, a fast stabilizer circuit simulator. "
            "It can be installed with:\n\npip install stim"
        ) from Error
    return stim


class DefaultClifford(Device):
    r"""A PennyLane device for fast simulation of Clifford circuits using
    `stim <https://github.com/quantumlib/stim/>`_.

    Args:
        wires (int, Iterable[Number, str]): Number of wires present on the device, or iterable that
            contains unique labels for the wires as numbers (i.e., ``[-1, 0, 2]``) or strings
            (``['ancilla', 'q1', 'q2']``). Default ``None`` if not specified.
        shots (int, Sequence[int], Sequence[Union[int, Sequence[int]]]): The default number of shots to use in executions involving
            this device.
        check_clifford (bool): Check if all the gate operations in the circuits to be executed are Clifford. Default is ``True``.
        tableau (bool): Determines what should be returned when the device's state is computed with :func:`qml.state <pennylane.state>`.
            When ``True``, the device returns the final evolved Tableau. Alternatively, one may make it ``False`` to obtain
            the evolved state vector. Note that the latter might not be computationally feasible for larger qubit numbers.
        seed (Union[str, None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``, or
            a request to seed from numpy's global random number generator.
            The default, ``seed="global"`` pulls a seed from numpy's global generator. ``seed=None``
            will pull a seed from the OS entropy.
        max_workers (int): A ``ProcessPoolExecutor`` executes tapes asynchronously
            using a pool of at most ``max_workers`` processes. If ``max_workers`` is ``None``,
            only the current process executes tapes. If you experience any
            issue, try setting ``max_workers`` to ``None``.

    **Example:**

    .. code-block:: python

        dev = qml.device("default.clifford", tableau=True)

        @qml.qnode(dev)
        def circuit():
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=[1])
            qml.ISWAP(wires=[0, 1])
            qml.Hadamard(wires=[0])
            return qml.state()

    >>> circuit()
    array([[0, 1, 1, 0, 0],
            [1, 0, 1, 1, 1],
            [0, 0, 0, 1, 0],
            [1, 0, 0, 1, 1]])

    The devices execution pipeline can be investigated more closely with the following:

    .. code-block:: python

        num_qscripts = 5

        qscripts = [
            qml.tape.QuantumScript(
                [qml.Hadamard(wires=[0]), qml.CNOT(wires=[0, 1])],
                [qml.expval(qml.PauliZ(0))]
            )
        ] * num_qscripts

    >>> dev = DefaultClifford()
    >>> program, execution_config = dev.preprocess()
    >>> new_batch, post_processing_fn = program(qscripts)
    >>> results = dev.execute(new_batch, execution_config=execution_config)
    >>> post_processing_fn(results)
    (array(0), array(0), array(0), array(0), array(0))

    .. details::
        :title: Clifford Tableau
        :href: clifford-tableau-theory

        The device's internal state is represented by the following ``Tableau`` described in
        the `Sec. III, Aaronson & Gottesman (2004) <https://arxiv.org/abs/quant-ph/0406196>`_:

        .. math::

            \begin{bmatrix}
            x_{11} & \cdots & x_{1n} &        & z_{11} & \cdots & z_{1n} & &r_{1}\\
            \vdots & \ddots & \vdots & & \vdots & \ddots & \vdots & &\vdots\\
            x_{n1} & \cdots & x_{nn} &        & z_{n1} & \cdots & z_{nn} & &r_{n}\\
            & & & & & & & & \\
            x_{\left(  n+1\right)  1} & \cdots & x_{\left(  n+1\right)  n} & &
            z_{\left(  n+1\right)  1} & \cdots & z_{\left(  n+1\right)  n} & & r_{n+1}\\
            \vdots & \ddots & \vdots  & & \vdots & \ddots & \vdots & & \vdots\\
            x_{\left(  2n\right)  1}  & \cdots & x_{\left(  2n\right)  n} & &
            z_{\left(  2n\right)  1}  & \cdots & z_{\left(  2n\right)  n} & & r_{2n}
            \end{bmatrix}

        The tableau's first `n` rows represent a destabilizer generator, while the
        remaining `n` rows represent the stabilizer generators. The Pauli representation
        for all of these generators are described using the :mod:`binary vector <pennylane.pauli.binary_to_pauli>`
        made from the binary variables :math:`x_{ij},\ z_{ij}`,
        :math:`\forall i\in\left\{1,\ldots,2n\right\}, j\in\left\{1,\ldots,n\right\}`
        and they together form the complete Pauli group.

        Finally, the last column of the tableau, with binary variables
        :math:`r_{i},\ \forall i\in\left\{1,\ldots,2n\right\}`,
        denotes whether the phase is negative (:math:`r_i = 1`) or not, for each generator.
        Maintaining and working with this tableau representation instead of the complete state vector
        makes the calculations of increasingly large Clifford circuits more efficient on this device.

    .. details::
        :title: Probabilities for Basis States
        :href: clifford-probabilities

        One can compute the analytical probability of each computational basis state using
        :func:`qml.probs <pennylane.probs>`, which also accepts a wire specification for obtaining marginal
        probabilities and an observable for obtaining probabilities from the rotated computational basis.
        As the ``default.clifford`` device supports executingm quantum circuits with a large number of qubits,
        we restrict the ability to compute the probabilities for `all` computational basis states at once to
        maintain computational efficiency. In contrast, one can specify target state(s), i.e., subset
        `basis states` of interest, using the ``probability_target`` property of the device.

        .. code-block:: python

            import pennylane as qml
            import numpy as np

            dev = qml.device("default.clifford")
            dev.probability_target = np.array([[0, 0], [1, 0]])

        After doing this, one can simply use the :func:`qml.probs <pennylane.probs>` with its usual arguments
        and probabilities for the specified target states would be computed and returned.

        .. code-block:: python

            wires = np.random.randint(3, size=(10000, 3))

            @qml.qnode(dev)
            def circuit():
                for w in wires:
                    qml.PauliX(w[0])
                    qml.PauliY(w[1])
                    qml.PauliZ(w[2])
                return qml.probs(op = qml.PauliX(0) @ qml.PauliY(1))

        >>> circuit()
        tensor([0.25, 0.25], requires_grad=True)

        .. note::

            If there's a mismatch in the number of wires for the provided target state(s) and the observable,
            then the marginal probabilities will be returned for the computational basis states for the
            subspace built using minimum number of wires among the two.

    .. details::
        :title: Tracking
        :href: clifford-tracking

        ``DefaultClifford`` tracks:

        * ``executions``: the number of unique circuits that would be required on quantum hardware
        * ``shots``: the number of shots
        * ``resources``: the :class:`~.resource.Resources` for the executed circuit.
        * ``simulations``: the number of simulations performed. One simulation can cover multiple QPU executions,
          such as for non-commuting measurements and batched parameters.
        * ``batches``: The number of times :meth:`~.execute` is called.
        * ``results``: The results of each call of :meth:`~.execute`.

    .. details::
        :title: Accelerate calculations with multiprocessing
        :href: clifford-multiprocessing

        See the details in :class:`~pennylane.devices.DefaultQubit`'s "Accelerate calculations with multiprocessing"
        section. Additional information regarding multiprocessing can be found in the
        `multiprocessing docs page <https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods>`_.
    """

    @property
    def name(self):
        """The name of the device."""
        return "default.clifford"

    # pylint:disable = too-many-arguments
    def __init__(
        self,
        wires=None,
        shots=None,
        check_clifford=True,
        tableau=True,
        seed="global",
        max_workers=None,
    ) -> None:
        super().__init__(wires=wires, shots=shots)

        self._max_workers = max_workers
        self._check_clifford = check_clifford

        self._tableau = tableau
        self._prob_states = None

        self._seed = np.random.randint(0, high=10000000) if seed == "global" else seed
        self._rng = np.random.default_rng(self._seed)
        self._debugger = None

    def _setup_execution_config(self, execution_config: ExecutionConfig) -> ExecutionConfig:
        """This is a private helper for ``preprocess`` that sets up the execution config.

        Args:
            execution_config (ExecutionConfig)

        Returns:
            ExecutionConfig: a preprocessed execution config

        """

        updated_values = {}
        if execution_config.gradient_method == "best":  # pragma: no cover
            updated_values["gradient_method"] = None
        updated_values["use_device_jacobian_product"] = False
        if execution_config.grad_on_execution is None:
            updated_values["grad_on_execution"] = False
        updated_values["device_options"] = dict(execution_config.device_options)  # copy
        if "max_workers" not in updated_values["device_options"]:
            updated_values["device_options"]["max_workers"] = self._max_workers
        if "rng" not in updated_values["device_options"]:
            updated_values["device_options"]["rng"] = self._rng
        if "tableau" not in updated_values["device_options"]:
            updated_values["device_options"]["tableau"] = self._tableau
        return replace(execution_config, **updated_values)

    def preprocess(
        self,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Tuple[TransformProgram, ExecutionConfig]:
        """This function defines the device transform program to be applied and an updated device configuration.

        Args:
            execution_config (Union[ExecutionConfig, Sequence[ExecutionConfig]]): A data structure describing the
                parameters needed to fully describe the execution.

        Returns:
            TransformProgram, ExecutionConfig: A transform program that when called returns QuantumTapes that the device
            can natively execute as well as a postprocessing function to be called after execution, and a configuration with
            unset specifications filled in.

        This device:

        * Currently does not intrinsically support parameter broadcasting

        """
        config = self._setup_execution_config(execution_config)
        transform_program = TransformProgram()

        transform_program.add_transform(validate_device_wires, self.wires, name=self.name)
        transform_program.add_transform(qml.defer_measurements, device=self)

        # Perform circuit decomposition to the supported Clifford gate set
        if self._check_clifford:
            transform_program.add_transform(
                decompose, stopping_condition=operation_stopping_condition, name=self.name
            )
        transform_program.add_transform(
            validate_measurements, sample_measurements=accepted_sample_measurement, name=self.name
        )
        transform_program.add_transform(
            validate_observables, stopping_condition=observable_stopping_condition, name=self.name
        )

        # Validate multi processing
        max_workers = config.device_options.get("max_workers", self._max_workers)
        if max_workers:
            transform_program.add_transform(validate_multiprocessing_workers, max_workers, self)

        # Validate derivatives
        transform_program.add_transform(validate_adjoint_trainable_params)
        if config.gradient_method is not None:
            config.gradient_method = None

        return transform_program, config

    def execute(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Result_or_ResultBatch:
        is_single_circuit = False
        if isinstance(circuits, QuantumScript):
            is_single_circuit = True
            circuits = [circuits]

        max_workers = execution_config.device_options.get("max_workers", self._max_workers)
        if max_workers is None:
            seeds = self._rng.integers(2**31 - 1, size=len(circuits))
            results = tuple(
                self.simulate(c, seed=s, debugger=self._debugger) for c, s in zip(circuits, seeds)
            )
        else:
            vanilla_circuits = [convert_to_numpy_parameters(c) for c in circuits]
            seeds = self._rng.integers(2**31 - 1, size=len(vanilla_circuits))
            _wrap_simulate = partial(self.simulate, debugger=None)
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                exec_map = executor.map(_wrap_simulate, vanilla_circuits, seeds)
                results = tuple(exec_map)

            self._rng = np.random.default_rng(self._rng.integers(2**31 - 1))

        if self.tracker.active:
            self.tracker.update(batches=1)
            self.tracker.record()
            for i, c in enumerate(circuits):
                qpu_executions, shots = get_num_shots_and_executions(c)
                res = np.array(results[i]) if isinstance(results[i], Number) else results[i]
                if c.shots:  # pragma: no cover
                    self.tracker.update(
                        simulations=1,
                        executions=qpu_executions,
                        results=res,
                        shots=shots,
                        resources=c.specs["resources"],
                    )
                else:
                    self.tracker.update(
                        simulations=1,
                        executions=qpu_executions,
                        results=res,
                        resources=c.specs["resources"],
                    )
                self.tracker.record()

        return results[0] if is_single_circuit else results

    # pylint:disable=no-member
    def simulate(
        self,
        circuit: qml.tape.QuantumScript,
        seed=None,
        debugger=None,
    ) -> Result:
        """Simulate a single quantum script.

        Args:
            circuit (QuantumTape): The single circuit to simulate
            debugger (_Debugger): The debugger to use

        Returns:
            tuple(TensorLike): The results of the simulation

        This function assumes that all operations are Clifford.

        >>> qs = qml.tape.QuantumScript([qml.Hadamard(wires=0)], [qml.expval(qml.PauliZ(0)), qml.state()])
        >>> qml.devices.DefaultClifford().simulate(qs)
        (array(0),
         array([[0, 1, 0],
                [1, 0, 0]]))

        """

        stim = _import_stim()

        # Account for custom labelled wires
        circuit = circuit.map_to_standard_wires()

        # Build a stim circuit, tableau and simulator
        stim_ct = stim.Circuit()
        initial_tableau = stim.Tableau.from_circuit(stim_ct)
        tableau_simulator = stim.TableauSimulator()

        # Account for state preparation operation
        prep = None
        if len(circuit) > 0 and isinstance(circuit[0], qml.operation.StatePrepBase):
            prep = circuit[0]
        use_prep_ops = bool(prep)

        # TODO: Add a method to prepare directly from a Tableau
        if use_prep_ops:
            initial_tableau = stim.Tableau.from_state_vector(
                qml.math.reshape(prep.state_vector(wire_order=list(circuit.op_wires)), (1, -1))[0],
                endian="big",
            )
            tableau_simulator.do_tableau(initial_tableau, circuit.wires)

        # Iterate over the gates --> manage them manually or apply them to circuit
        global_phase_ops = []
        for op in circuit.operations[use_prep_ops:]:
            gate, wires = self._pl_to_stim(op)
            if gate is not None:
                # Note: This is a lot faster than doing `stim_ct.append(gate, wires)`
                stim_ct.append_from_stim_program_text(f"{gate} {wires}")
            else:
                if op.name == "GlobalPhase":
                    global_phase_ops.append(op)
                if op.name == "Snapshot":
                    if debugger is not None and debugger.active:
                        meas = op.hyperparameters["measurement"]
                        if meas is not None and not isinstance(meas, qml.measurements.StateMP):
                            raise ValueError(
                                f"{self.name} does not support arbitrary measurements of a state with snapshots."
                            )
                        state = stim.Tableau.from_circuit(stim_ct).to_state_vector()
                        if state.shape == (1,):
                            # following is faster than using np.eye(length=1, size, index)
                            state = qml.math.zeros(2**circuit.num_wires, dtype=complex)
                            state[0] = 1.0 + 0.0j
                        flat_state = qml.math.flatten(state)
                        debugger.snapshots[op.tag or len(debugger.snapshots)] = flat_state

        tableau_simulator.do_circuit(stim_ct)

        global_phase = qml.GlobalPhase(qml.math.sum(op.data[0] for op in global_phase_ops))

        # Perform measurments based on whether shots are provided
        if circuit.shots:
            stim_circuit = tableau_simulator.current_inverse_tableau().inverse().to_circuit()
            meas_results = self.measure_statistical(circuit, stim_circuit, stim, seed=seed)
        else:
            meas_results = self.measure_analytical(circuit, tableau_simulator, global_phase, stim)

        return meas_results[0] if len(meas_results) == 1 else tuple(meas_results)

    @staticmethod
    def _pl_to_stim(op):
        """Convert PennyLane operation to a Stim operation"""
        try:
            stim_op = _GATE_OPERATIONS[op.name]
        except KeyError as e:
            raise qml.DeviceError(
                f"Operator {op} not supported on default.clifford and does not provide a decomposition."
            ) from e
        return stim_op, " ".join(map(str, op.wires))

    # pylint:disable=too-many-return-statements
    def _convert_op_to_linear_comb(self, meas_obs, coeffs, paulis):
        """Convert a PennyLane observable to a linear combination of stim Pauli terms"""

        # Case for simple Pauli terms
        if isinstance(meas_obs, (qml.Identity, qml.PauliZ, qml.PauliX, qml.PauliY)):
            coeffs.append(1.0)
            paulis.append((_GATE_OPERATIONS[meas_obs.name], meas_obs.wires))
            return coeffs, paulis

        # Case for Sum
        if isinstance(meas_obs, Sum):
            for op in meas_obs:
                coeffs, paulis = self._convert_op_to_linear_comb(op, coeffs, paulis)
            return coeffs, paulis

        # Case for higher arithmetic depth for prod-type observables
        if meas_obs.arithmetic_depth > 1 and isinstance(meas_obs, (Prod, SProd)):
            with qml.operation.use_new_opmath():
                meas_obs_simp = meas_obs.simplify()

            # Recurse only if the simplification happened
            if meas_obs_simp != meas_obs:
                coeffs, paulis = self._convert_op_to_linear_comb(meas_obs_simp, coeffs, paulis)
                return coeffs, paulis

        # Case for Prod
        if isinstance(meas_obs, Prod):
            coeffs.append(1.0)
            paulis.append(
                (
                    "".join([_GATE_OPERATIONS[op.name] for op in meas_obs.operands]),
                    meas_obs.wires,
                )
            )
            return coeffs, paulis

        # Case for SProd
        if isinstance(meas_obs, SProd):
            cof, obs = meas_obs.terms()
            gate_ops = obs
            if len(gate_ops) == 1 and isinstance(gate_ops[0], Prod):
                gate_ops = obs[0].operands
            coeffs.extend(cof)
            paulis.append(("".join([_GATE_OPERATIONS[op.name] for op in gate_ops]), meas_obs.wires))
            return coeffs, paulis

        # Add support for more case when the time is right
        # TODO: A very limited support for Exp/Evolution could be added.
        raise NotImplementedError(
            f"default.clifford doesn't support expectation value calculation with {type(meas_obs)} at the moment."
        )

    def _measure_observable_sample(self, meas_obs, stim_circuit, shots, sample_seed):
        """Compute sample output from a stim circuit for a given Pauli observable"""
        meas_dict = {"X": "MX", "Y": "MY", "Z": "MZ", "_": "M"}

        meas_op = qml.operation.convert_to_opmath(meas_obs)
        coeffs, paulis = self._convert_op_to_linear_comb(meas_op, coeffs=[], paulis=[])

        samples = []
        for pauli, wire in paulis:
            stim_circ = stim_circuit.copy()
            for op, wr in zip(pauli, wire):
                if op != "I":
                    stim_circ.append(meas_dict[op], wr)
            sampler = stim_circ.compile_sampler(seed=sample_seed)
            samples.append(qml.math.array(sampler.sample(shots=shots), dtype=int))

        return samples, qml.math.array(coeffs)

    # pylint:disable=protected-access
    def measure_statistical(self, circuit, stim_circuit, stim, seed=None):
        """Given a circuit, compute samples and return the statistical measurement results."""
        # Compute samples via circuits from tableau
        num_shots = circuit.shots.total_shots
        sample_seed = seed if isinstance(seed, int) else self._seed

        results = []
        for meas in circuit.measurements:
            meas_op = meas.obs
            if meas_op is None:
                meas_op = qml.prod(
                    *[
                        qml.PauliZ(idx)
                        for idx in (meas.wires if meas.wires else range(stim_circuit.num_qubits))
                    ]
                )
            # Computing samples via stim's compiled sampler
            if isinstance(meas, SampleMP):
                results.append(
                    self._measure_observable_sample(meas_op, stim_circuit, num_shots, sample_seed)[
                        0
                    ][0]
                )
            # Computing classical shadows via manual single sampling
            elif isinstance(meas, ClassicalShadowMP):
                results.append(self._measure_classical_shadow(stim_circuit, circuit, meas, stim))

            # Computing observable expectation value using above classical shadows
            elif isinstance(meas, ShadowExpvalMP):
                results.append(self._measure_expval_shadow(stim_circuit, circuit, meas, stim))

            elif isinstance(meas, ExpectationMP):
                results.append(
                    self._measure_expectation_samples(stim_circuit, meas_op, num_shots, sample_seed)
                )

            elif isinstance(meas, VarianceMP):
                meas_obs = qml.operation.convert_to_opmath(meas_op)
                with qml.operation.use_new_opmath():
                    meas_obs1 = meas_obs.simplify()
                    meas_obs2 = (meas_obs1**2).simplify()

                # use the naive formula for variance, i.e., Var(Q) = ⟨𝑄^2⟩−⟨𝑄⟩^2
                vars = (
                    self._measure_expectation_samples(
                        stim_circuit, meas_obs2, num_shots, sample_seed
                    )
                    - self._measure_expectation_samples(
                        stim_circuit, meas_obs1, num_shots, sample_seed
                    )
                    ** 2
                )
                results.append(vars)

            # Computing rest of the measurement by processing samples
            else:
                samples = qml.math.array(
                    [
                        self._measure_observable_sample(
                            meas_op, stim_circuit, num_shots, sample_seed
                        )[0]
                    ]
                )
                results.append(meas.process_samples(samples=samples, wire_order=circuit.wires))
        return results

    def measure_analytical(self, circuit, tableau_simulator, global_phase, stim):
        """Given a circuit, compute tableau and return the analytical measurement results."""
        results = []
        for meas in circuit.measurements:
            # Computing density matrix via tableaus
            if isinstance(meas, DensityMatrixMP):  # do first because it is a child of StateMP
                res = self._measure_density_matrix(tableau_simulator, list(meas.wires))

            # Computing statevector via tableaus
            elif isinstance(meas, StateMP):
                res = self._measure_state(tableau_simulator, circuit, global_phase)

            # Computing expectation values via measurement
            elif isinstance(meas, ExpectationMP):
                res = self._measure_expectation(tableau_simulator, meas, stim)

            # Computing variance via measurement
            elif isinstance(meas, VarianceMP):
                res = self._measure_variance(tableau_simulator, meas, stim)

            # Computing probabilities via tableau
            elif isinstance(meas, ProbabilityMP):
                res = self._measure_probability(tableau_simulator, circuit, meas, stim)

            # Computing entropy via tableaus
            elif isinstance(meas, VnEntropyMP):
                tableau = tableau_simulator.current_inverse_tableau().inverse()
                z_stabs = qml.math.array(
                    [tableau.z_output(wire) for wire in range(len(circuit.wires))]
                )
                res = self._measure_stabilizer_vn_entropy(z_stabs, list(meas.wires))

            # Computing mutual-info via tableaus
            elif isinstance(meas, MutualInfoMP):
                tableau = tableau_simulator.current_inverse_tableau().inverse()
                z_stabs = qml.math.array(
                    [tableau.z_output(wire) for wire in range(len(circuit.wires))]
                )
                indices0, indices1 = getattr(meas, "_wires")
                res = self._measure_stabilizer_vn_entropy(
                    z_stabs, list(indices0)
                ) + self._measure_stabilizer_vn_entropy(z_stabs, list(indices1))

            # Computing purity via tableaus
            elif isinstance(meas, PurityMP):
                tableau = tableau_simulator.current_inverse_tableau().inverse()
                z_stabs = qml.math.array(
                    [tableau.z_output(wire) for wire in range(len(circuit.wires))]
                )
                res = (
                    qml.math.array(1.0)
                    if circuit.op_wires == meas.wires
                    else self._measure_purity(z_stabs, list(meas.wires))
                )

            # Computing more measurements
            else:  # pragma: no cover
                raise NotImplementedError(
                    f"default.clifford doesn't support the {type(meas)} measurement at the moment."
                )

            results.append(res)

        return results

    def _measure_state(self, tableau_simulator, circuit, global_phase):
        """Measure the state of the simualtor device."""
        if self._tableau:
            # Stack according to Sec. III, arXiv:0406196 (2008)
            tableau = tableau_simulator.current_inverse_tableau().inverse()
            x2x, x2z, z2x, z2z, x_signs, z_signs = tableau.to_numpy()
            pl_tableau = np.vstack(
                (
                    np.hstack((x2x, x2z, x_signs.reshape(-1, 1))),
                    np.hstack((z2x, z2z, z_signs.reshape(-1, 1))),
                )
            ).astype(int)
            if pl_tableau.shape == (0, 1) and circuit.num_wires:
                return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            return pl_tableau

        state = qml.math.array(tableau_simulator.state_vector(endian="big"))
        if state.shape == (1,) and circuit.num_wires:
            # following is faster than using np.eye(length=1, size, index)
            state = qml.math.zeros(2**circuit.num_wires, dtype=complex)
            state[0] = 1.0 + 0.0j
        return state * qml.matrix(global_phase)

    @staticmethod
    def _measure_density_matrix(tableau_simulator, wires):
        """Measure the density matrix from the state of simulator device."""
        state_vector = qml.math.array(tableau_simulator.state_vector(endian="big"))
        return qml.math.reduce_dm(qml.math.einsum("i, j->ij", state_vector, state_vector), wires)

    def _measure_expectation(self, tableau_simulator, meas_op, stim):
        """Measure the expectation value with respect to the state of simulator device."""
        # Get the observable for the expectation value measurement
        meas_obs = qml.operation.convert_to_opmath(meas_op.obs)
        coeffs, paulis = self._convert_op_to_linear_comb(meas_obs, coeffs=[], paulis=[])

        expecs = qml.math.zeros_like(coeffs)
        for idx, (pauli, wire) in enumerate(paulis):
            pauli_term = ["I"] * max(np.max(list(wire)) + 1, tableau_simulator.num_qubits)
            for op, wr in zip(pauli, wire):
                pauli_term[wr] = op
            stim_pauli = stim.PauliString("".join(pauli_term))
            expecs[idx] = tableau_simulator.peek_observable_expectation(stim_pauli)

        return qml.math.dot(coeffs, expecs)

    def _measure_expectation_samples(self, stim_circuit, meas_op, num_shots, sample_seed):
        """Measure the expectation value with respect to samples from simulator device."""
        # Get the observable for the expectation value measurement
        samples, coeffs = self._measure_observable_sample(
            meas_op, stim_circuit, num_shots, sample_seed
        )
        expecs = [
            qml.math.mean(qml.math.power([-1] * num_shots, qml.math.sum(sample, axis=1)))
            for sample in samples
        ]

        return qml.math.dot(coeffs, expecs)

    def _measure_variance(self, tableau_simulator, meas_op, stim):
        """Measure the variance with respect to the state of simulator device."""
        meas_obs = qml.operation.convert_to_opmath(meas_op.obs)
        with qml.operation.use_new_opmath():
            meas_obs1 = meas_obs.simplify()
            meas_obs2 = (meas_obs1**2).simplify()

        # use the naive formula for variance, i.e., Var(Q) = ⟨𝑄^2⟩−⟨𝑄⟩^2
        return (
            self._measure_expectation(tableau_simulator, ExpectationMP(meas_obs2), stim)
            - self._measure_expectation(tableau_simulator, ExpectationMP(meas_obs1), stim) ** 2
        )

    # pylint: disable=protected-access
    @staticmethod
    def _measure_stabilizer_vn_entropy(stabilizer, wires, log_base=None):
        r"""Computes the Rényi entanglement entropy using stabilizer information.

        Computes the Rényi entanglement entropy :math:`S_A` for a subsytem described by
        :math:`A`, :math:`S_A = \text{rank}(\text{projection}_A {\mathcal{S}}) - |\mathcal{S}|`,
        where :math:`\mathcal{S}` is the stabilizer group for the system using the theory
        described in `arXiv:1901.08092 <https://arxiv.org/abs/1901.08092>`_.

        Args:
            stabilizer (TensorLike): stabilizer set for the system
            wires (Iterable): wires describing the subsystem
            log_base (int): base for the logarithm.

        Returns:
            (float): entanglement entropy of the subsystem
        """
        # Get the number of qubits for the system
        num_qubits = qml.math.shape(stabilizer)[0]

        # Von Neumann entropy of a stabilizer state is zero
        if len(wires) == num_qubits:
            return 0.0

        # Build a binary matrix desribing the stabilizers using the Pauli words
        pauli_dict = {0: "I", 1: "X", 2: "Y", 3: "Z"}
        terms = [
            qml.pauli.PauliWord({idx: pauli_dict[ele] for idx, ele in enumerate(row)})
            for row in stabilizer
        ]
        binary_mat = qml.pauli.utils._binary_matrix_from_pws(terms, num_qubits)

        # Partition the binary matrix to represent the subsystem
        partition_mat = qml.math.hstack(
            (
                binary_mat[:, num_qubits:][:, wires],
                binary_mat[:, :num_qubits][:, wires],
            )
        )

        # Use the reduced row echelon form for finding rank efficiently
        # tapering always come in handy :)
        rank = qml.math.sum(
            qml.math.any(qml.qchem.tapering._reduced_row_echelon(partition_mat), axis=1)
        )

        # Use the Eq. A17 in arXiv:1901.08092 for entropy calculation
        entropy = qml.math.log(2) * (rank - len(wires))

        # Determine wether to use any log base
        if log_base is None:
            return entropy

        return entropy / qml.math.log(log_base)

    def _measure_purity(self, stabilizer, wires):
        r"""Measure the purity of the state of simulator device.

        Computes the state purity using the monotonically decreasing second-order Rényi entropy
        form given in `Sci Rep 13, 4601 (2023) <https://www.nature.com/articles/s41598-023-31273-9>`_.
        We utilize the fact that Rényi entropies are equal for all Rényi indices ``n`` for the
        stabilizer states.

        Args:
            stabilizer (TensorLike): stabilizer set for the system
            wires (Iterable): wires describing the subsystem
            log_base (int): base for the logarithm.

        Returns:
            (float): entanglement entropy of the subsystem
        """
        return 2 ** (-self._measure_stabilizer_vn_entropy(stabilizer, wires, log_base=2))

    @property
    def probability_target(self):
        """Get the target computational basis states for computing outcome probability."""
        return self._prob_states

    @probability_target.setter
    def probability_target(self, basis_states):
        """Set the target computational basis states for computing outcome probability."""
        self._prob_states = qml.math.stack(
            basis_states if len(qml.math.shape(basis_states)) > 1 else [basis_states]
        )

    # pylint: disable=too-many-branches
    def _measure_probability(self, tableau_simulator, circuit, meas_op, stim):
        """Measure the probability of each computational basis state."""

        if self._prob_states is None and self._tableau:
            raise ValueError(
                "In order to maintain computational efficiency, \
                with ``tableau=True``, the clifford device supports returning \
                probability only for selected target computational basis states. \
                Please use the `probability_target` property to set them."
            )

        # TODO: We might be able to skip the inverse done below
        # (as the distribution should be independent of inverse)
        diagonalizing_cit = tableau_simulator.current_inverse_tableau().inverse().to_circuit()
        diagonalizing_ops = [] if not meas_op.obs else meas_op.obs.diagonalizing_gates()
        for diag_op in diagonalizing_ops:
            # Check if it is Clifford
            if diag_op.name not in _GATE_OPERATIONS:  # pragma: no cover
                raise ValueError(
                    f"Currently, we only support observables whose diagonalizing gates are Clifford, got {diag_op}"
                )
            # Add to the circuit to rotate the basis
            stim_op = self._pl_to_stim(diag_op)
            if stim_op[0] is not None:
                diagonalizing_cit.append_from_stim_program_text(f"{stim_op[0]} {stim_op[1]}")

        # Build the Tableau simulator from the diagonalized circuit
        circuit_simulator = stim.TableauSimulator()
        circuit_simulator.do_circuit(diagonalizing_cit)
        if not self._tableau:
            state = self._measure_state(circuit_simulator, circuit, qml.GlobalPhase(0.0))
            return meas_op.process_state(state, wire_order=circuit.wires)

        # Obtain the measurement wires for getting the basis states
        mobs_wires = meas_op.obs.wires if meas_op.obs else meas_op.wires
        meas_wires = mobs_wires if mobs_wires else circuit.wires
        tgt_states = self._prob_states
        if not tgt_states.shape[1]:
            raise ValueError("Cannot set an empty list of target states.")

        if len(meas_wires) <= tgt_states.shape[1]:
            tgt_states = []
            for state in self._prob_states:
                if list(state[meas_wires]) not in tgt_states:
                    tgt_states.append(list(state[meas_wires]))
            tgt_states = np.array(tgt_states)
        else:
            meas_wires = meas_wires[: tgt_states.shape[1]]

        # Iterate over the measured qubits and post-select possible outcome
        # This should now scaled as O(M * N), where N is the number of measured qubits,
        # and M is the cost of peeking and postselection of each qubit in computational basis.
        prob_res = np.ones(tgt_states.shape[0])
        for wire in meas_wires:
            expectation = circuit_simulator.peek_z(wire)
            # (Eig --> Res) | -1 --> 1 | 1 --> 0 | 0 --> 0 / 1 |
            outcome = int(0.5 * (1 - expectation))
            if not expectation:
                prob_res /= 2.0
            else:
                prob_res[np.where(outcome != tgt_states[:, wire])[0]] = 0.0
            circuit_simulator.postselect_z(wire, desired_value=outcome)

        return prob_res

    @staticmethod
    def _measure_single_sample(stim_ct, meas_ops, meas_idx, meas_wire, stim):
        """Sample a single qubit Pauli measurement from a stim circuit"""
        stim_sm = stim.TableauSimulator()
        stim_sm.do_circuit(stim_ct)
        return stim_sm.measure_observable(
            stim.PauliString([0] * meas_idx + meas_ops + [0] * (meas_wire - meas_idx - 1))
        )

    def _measure_classical_shadow(self, stim_circuit, circuit, meas_op, stim):
        """Measures classical shadows from the state of simulator device"""
        meas_seed = meas_op.seed or np.random.randint(2**30)
        meas_wire = len(circuit.wires)

        bits = []
        recipes = np.random.RandomState(meas_seed).randint(
            3, size=(circuit.shots.total_shots, meas_wire)
        )  # Random Pauli basis to be used for measurements

        for recipe in recipes:
            bits.append(
                [
                    self._measure_single_sample(stim_circuit, [rec + 1], idx, meas_wire, stim)
                    for idx, rec in enumerate(recipe)
                ]
            )

        return np.asarray(bits, dtype=int), np.asarray(recipes, dtype=int)

    def _measure_expval_shadow(self, stim_circuit, circuit, meas_op, stim):
        """Measures expectation value of a Pauli observable using
        classical shadows from the state of simulator device."""
        bits, recipes = self._measure_classical_shadow(stim_circuit, circuit, meas_op, stim)
        # TODO: Benchmark scaling for larger number of circuits for this existing functionality
        shadow = qml.shadows.ClassicalShadow(bits, recipes, wire_map=circuit.wires.tolist())
        return shadow.expval(meas_op.H, meas_op.k)
