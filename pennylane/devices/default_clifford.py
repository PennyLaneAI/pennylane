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
from pennylane import DeviceError
from pennylane.tape import QuantumTape, QuantumScript
from pennylane.typing import Result, ResultBatch
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.transforms.core import TransformProgram
from pennylane.measurements import ExpectationMP, StateMP, DensityMatrixMP, PurityMP
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
    "Exp",
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

        seed = np.random.randint(0, high=10000000) if seed == "global" else seed
        self._rng = np.random.default_rng(seed)
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

        * Currently does not support finite shots
        * Currently does not intrinsically support parameter broadcasting

        """
        config = self._setup_execution_config(execution_config)
        transform_program = TransformProgram()

        transform_program.add_transform(validate_device_wires, self.wires, name=self.name)
        transform_program.add_transform(qml.defer_measurements, device=self)

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
            results = tuple(self.simulate(c, debugger=self._debugger) for c in circuits)
        else:
            vanilla_circuits = [convert_to_numpy_parameters(c) for c in circuits]
            _wrap_simulate = partial(self.simulate, debugger=None)
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                exec_map = executor.map(_wrap_simulate, vanilla_circuits)
                results = tuple(exec_map)

            # reset _rng to mimic serial behavior - TODO: uncomment when using RNG
            # self._rng = np.random.default_rng(self._rng.integers(2**31 - 1))

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

        if circuit.shots:
            raise NotImplementedError(
                "default.clifford currently doesn't support computation with shots."
            )

        # Build a stim circuit, tableau and simulator
        stim_ct = stim.Circuit()
        initial_tableau = stim.Tableau.from_circuit(stim_ct)
        tableau_simulator = stim.TableauSimulator()

        # Account for state preparation operation
        prep = None
        if len(circuit) > 0 and isinstance(circuit[0], qml.operation.StatePrepBase):
            prep = circuit[0]
        use_prep_ops = bool(prep)

        if use_prep_ops:
            initial_tableau = stim.Tableau.from_state_vector(
                qml.math.reshape(prep.state_vector(wire_order=list(circuit.op_wires)), (1, -1))[0],
                endian="big",
            )
            tableau_simulator.do_tableau(initial_tableau, circuit.wires)

        global_phase_ops = []
        for op in circuit.operations[use_prep_ops:]:
            gate, wires = self.pl_to_stim(op)
            if gate is not None:
                # Note: This is ~300x faster than doing stim_ct.append(gate, wires)
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
        return self.measure(circuit, tableau_simulator, global_phase, stim)

    @staticmethod
    def pl_to_stim(op):
        """Convert PennyLane operation to a Stim operation"""
        try:
            stim_op = _GATE_OPERATIONS[op.name]
        except KeyError as e:
            raise DeviceError(
                f"Operator {op} not supported on default.clifford and does not provide a decomposition."
            ) from e
        return stim_op, " ".join(map(str, op.wires))

    def measure(self, circuit, tableau_simulator, global_phase, stim):
        """Given a circuit, compute and return the measurement results."""
        results = []
        for meas in circuit.measurements:
            # Analytic case
            if not circuit.shots:
                # Computing density matrix via tableaus
                if isinstance(meas, DensityMatrixMP):  # do first because it is a child of StateMP
                    res = self._measure_density_matrix(tableau_simulator)

                # Computing statevector via tableaus
                elif isinstance(meas, StateMP):
                    res = self._measure_state(tableau_simulator, circuit, global_phase)

                # Computing purity via tableaus
                elif isinstance(meas, PurityMP):
                    res = self._measure_purity(meas, circuit)

                # Computing expectation values via measurement
                elif isinstance(meas, ExpectationMP):
                    res = self._measure_expectation(tableau_simulator, meas, stim)

                # Computing more measurements
                else:
                    raise NotImplementedError(
                        f"default.clifford doesn't support the {type(meas)} measurement at the moment."
                    )

                results.append(res)

        return results[0] if len(results) == 1 else tuple(results)

    def _measure_state(self, tableau_simulator, circuit, global_phase):
        """Measure the state of the simualtor device."""
        if self._tableau:
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
    def _measure_density_matrix(tableau_simulator):
        """Measure the density matrix from the state of simulator device."""
        state_vector = qml.math.array(tableau_simulator.state_vector(endian="big"))
        return qml.math.einsum("i, j->ij", state_vector, state_vector)

    @staticmethod
    def _measure_purity(meas_op, circuit):
        """Measure the purity of the state of simulator device"""
        if circuit.wires != meas_op.wires:
            raise NotImplementedError(
                "default.clifford doesn't support measuring the purity of a subset of wires at the moment."
            )
        return qml.math.array(1.0)  # // Trivial

    @staticmethod
    def _measure_expectation(tableau_simulator, meas_op, stim):
        """Measure the expectation value with respect to the state of simulator device."""

        # Case for simple Pauli terms
        if isinstance(meas_op.obs, (qml.PauliZ, qml.PauliX, qml.PauliY)):
            pauli = stim.PauliString(_GATE_OPERATIONS[meas_op.obs.name])
            return qml.math.array(tableau_simulator.peek_observable_expectation(pauli))

        # Case for simple Pauli tensor
        if isinstance(meas_op.obs, qml.operation.Tensor):
            expec = "".join([_GATE_OPERATIONS[name] for name in meas_op.obs.name])
            pauli = stim.PauliString(expec)
            return qml.math.array(tableau_simulator.peek_observable_expectation(pauli))

        # Case for a Hamiltonian
        if isinstance(meas_op.obs, qml.Hamiltonian):
            coeffs, obs = meas_op.obs.terms()
            expecs = qml.math.zeros_like(coeffs)
            for idx, ob in enumerate(obs):
                expec = "".join([_GATE_OPERATIONS[name] for name in ob.name])
                pauli = stim.PauliString(expec)
                expecs[idx] = tableau_simulator.peek_observable_expectation(pauli)
            return qml.math.dot(coeffs, expecs)

        # Add support for more case when the time is right
        raise NotImplementedError(
            f"default.clifford doesn't support expectation value calculation with {type(meas_op.obs)} at the moment."
        )
