# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
This module contains the clifford simulator based on stim
"""

from dataclasses import replace
from functools import partial
from numbers import Number
from typing import Union, Tuple, Optional, Sequence
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
    warn_about_trainable_observables,
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
    """Specifies whether or not an operation is accepted by DefaultClifford."""
    return op.name in _GATE_OPERATIONS


def observable_stopping_condition(obs: qml.operation.Operator) -> bool:
    """Specifies whether or not an observable is accepted by DefaultClifford."""
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
    r"""A PennyLane device written in Python and capable of executing Clifford circuits using
    `stim (2021) <https://github.com/quantumlib/stim/tree/main>`_.

    Args:
        shots (int, Sequence[int], Sequence[Union[int, Sequence[int]]]): The default number of shots to use in executions involving
            this device.
        check_clifford (bool): Check if all the gate operations in the circuits to be executed are Clifford. Default is ``True``.
        max_error (float): The maximum permissible operator norm error for decomposing circuits with non-Clifford gate operations
            into Clifford+T basis. The default is ``None`` as this device currently supports only Clifford simulations.
        tableau (bool): Determines what should be returned when the device's state is computed with ``qml.state``. Default is
            ``True``, which makes it return the final evolved Tableau. Alternatively, one may make it ``False`` to obtain
            the evolved state vector. Note that the latter might not be computationally feasible for larger qubit numbers.
        seed (Union[str, None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``, or
            a request to seed from numpy's global random number generator.
            The default, ``seed="global"`` pulls a seed from NumPy's global generator. ``seed=None``
            will pull a seed from the OS entropy.
        max_workers (int): A ``ProcessPoolExecutor`` executes tapes asynchronously
            using a pool of at most ``max_workers`` processes. If ``max_workers`` is ``None``,
            only the current process executes tapes. If you experience any
            issue, try setting ``max_workers`` to ``None``.

    **Example:**

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

        The device represents a state by the inverse of a Tableau
        (`Aaronson & Gottesman (2004) <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.70.052328>`_)
        consisiting of binary variable :math:`x_{ij},\ z_{ij}` for all :math:`i\in\left\{1,\ldots,2n\right\}`,
        :math:`j\in\left\{1,\ldots,n\right\}`, and :math:`r_{i}` for all
        :math:`i\in\left\{1,\ldots,2n\right\}`.

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

        Rows :math:`1` to :math:`n` of the tableau represent the destabilizer generators
        :math:`R_{1},\ldots,R_{n}`, and rows :math:`n+1` to :math:`2n` represent the stabilizer
        generators :math:`R_{n+1},\ldots,R_{2n}`. If :math:`R_{i}=\pm P_{1}\ldots P_{n}`,
        then bits :math:`x_{ij},z_{ij}` determine the j\ :math:`^{th}` Pauli matrix
        :math:`P_{j}:\ 00` means ``I``, :math:`01` means ``X``, :math:`11` means ``Y``,
        and :math:`10` means ``Z``. Finally, :math:`r_{i}` is one if :math:`R_{i}`
        has negative phase and zero if :math:`r_{i}` has a positive phase.


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
        * ``results``: The results of each call of :meth:`~.execute`
        * ``derivative_batches``: How many times :meth:`~.compute_derivatives` is called.
        * ``execute_and_derivative_batches``: How many times :meth:`~.execute_and_compute_derivatives` is called
        * ``derivatives``: How many circuits are submitted to :meth:`~.compute_derivatives` or
          :meth:`~.execute_and_compute_derivatives`.

    .. details::
        :title: Accelerate calculations with multiprocessing
        :href: clifford-multiprocessing

        Suppose one has a processor with 5 cores or more, these scripts can be executed in
        parallel as follows

        >>> dev = DefaultClifford(max_workers=5)
        >>> program, execution_config = dev.preprocess()
        >>> new_batch, post_processing_fn = program(qscripts)
        >>> results = dev.execute(new_batch, execution_config=execution_config)
        >>> post_processing_fn(results)

        If you monitor your CPU usage, you should see 5 new Python processes pop up to
        crunch through those ``QuantumScript``'s. Beware not oversubscribing your machine.
        This may happen if a single device already uses many cores, if NumPy uses a multi-
        threaded BLAS library like MKL or OpenBLAS for example. The number of threads per
        process times the number of processes should not exceed the number of cores on your
        machine. You can control the number of threads per process with the environment
        variables:

        * ``OMP_NUM_THREADS``
        * ``MKL_NUM_THREADS``
        * ``OPENBLAS_NUM_THREADS``

        where the last two are specific to the MKL and OpenBLAS libraries specifically.

        .. warning::

            Multiprocessing may fail depending on your platform and environment (Python shell,
            script with a protected entry point, Jupyter notebook, etc.) This may be solved
            changing the so-called start method. The supported start methods are the following:

            * Windows (win32): spawn (default).
            * macOS (darwin): spawn (default), fork, forkserver.
            * Linux (unix): spawn, fork (default), forkserver.

            which can be changed with ``multiprocessing.set_start_method()``. For example,
            if multiprocessing fails on macOS in your Jupyter notebook environment, try
            restarting the session and adding the following at the beginning of the file:

            .. code-block:: python

                import multiprocessing
                multiprocessing.set_start_method("fork")

            Additional information can be found in the
            `multiprocessing doc <https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods>`_.


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
        max_error=None,
        tableau=True,
        seed="global",
        max_workers=None,
    ) -> None:
        super().__init__(wires=wires, shots=shots)

        self._max_workers = max_workers
        self._check_clifford = check_clifford

        if max_error is None:
            self._max_error = 0.0
        else:
            raise ValueError(
                f"Currently this device doesn't support executing non-Clifford operations, got max_error={max_error}."
            )

        self._tableau = tableau
        self._prob_states = None

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
        if execution_config.gradient_method == "device":
            updated_values["use_device_gradient"] = True
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
        if "max_error" not in updated_values["device_options"]:
            updated_values["device_options"]["max_error"] = self._max_error
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
                decompose, stopping_condition=operation_stopping_condition, name="default.clifford"
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
        transform_program.add_transform(warn_about_trainable_observables)
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

    def supports_derivatives(
        self,
        execution_config: Optional[ExecutionConfig] = None,
        circuit: Optional[QuantumTape] = None,
    ) -> bool:
        """Check whether or not derivatives are available for a given configuration and circuit.

        ``DefaultClifford`` returns trivial derivates everytime.

        Args:
            execution_config (ExecutionConfig): The configuration of the desired derivative calculation
            circuit (QuantumTape): An optional circuit to check derivatives support for.

        Returns:
            Bool: Whether or not a derivative can be calculated provided the given information

        """

        return True

    def compute_derivatives(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        is_single_circuit = False
        if isinstance(circuits, QuantumScript):
            is_single_circuit = True
            circuits = [circuits]

        if self.tracker.active:
            self.tracker.update(derivative_batches=1, derivatives=len(circuits))
            self.tracker.record()

        res = tuple(tuple(np.zeros(s) for s in circuit.shape(self)) for circuit in circuits)

        return res[0] if is_single_circuit else res

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

        >>> qs = qml.tape.QuantumScript([qml.Hadamard(1.2, wires=0)], [qml.expval(qml.PauliZ(0)), qml.state(wires=(0,1))])
        >>> simulate(qs)
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
                stim_ct.append(gate, wires)
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

        if circuit.shots:
            stim_circuit = tableau_simulator.current_inverse_tableau().inverse().to_circuit()
            meas_results = self.measure_statistical(circuit, stim_circuit, stim, rng=None)
        else:
            meas_results = self.measure_analytical(circuit, tableau_simulator, global_phase, stim)

        return meas_results[0] if len(meas_results) == 1 else tuple(meas_results)

    @staticmethod
    def pl_to_stim(op):
        """Convert PennyLane operation to a Stim operation"""
        return _GATE_OPERATIONS[op.name], op.wires

    # pylint:disable=protected-access
    def measure_statistical(self, circuit, stim_circuit, stim, rng=None):
        """Given a circuit, compute samples and return the statistical measurement results."""
        # Compute samples via circuits from tableau
        sample_seed = rng if isinstance(rng, int) else self._seed
        stim_circ = stim_circuit.copy()
        stim_circ.append("M", circuit.wires)
        sampler = stim_circ.compile_sampler(seed=sample_seed)
        samples = qml.math.array(sampler.sample(shots=circuit.shots.total_shots), dtype=int)

        results = []
        for meas in circuit.measurements:
            # Computing samples via stim's compiled sampler
            if isinstance(meas, SampleMP):
                results.append(samples)

            # Computing classical shadows via manual single sampling
            elif isinstance(meas, ClassicalShadowMP):
                results.append(self._measure_classical_shadow(stim_circuit, circuit, meas, stim))

            # Computing observable expectation value using above classical shadows
            elif isinstance(meas, ShadowExpvalMP):
                results.append(self._measure_expval_shadow(stim_circuit, circuit, meas, stim))

            # Computing rest of the measurement by processing samples
            else:
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
                tableau = tableau_simulator.current_inverse_tableau() ** -1
                zs = qml.math.array([tableau.z_output(k) for k in range(len(circuit.wires))])
                res = self._stabilizer_vn_entropy(zs, list(meas.wires))

            # Computing mutual-info via tableaus
            elif isinstance(meas, MutualInfoMP):
                tableau = tableau_simulator.current_inverse_tableau() ** -1
                zs = qml.math.array([tableau.z_output(k) for k in range(len(circuit.wires))])
                indices0, indices1 = getattr(meas, "_wires")
                res = self._stabilizer_vn_entropy(zs, list(indices0)) + self._stabilizer_vn_entropy(
                    zs, list(indices1)
                )

            # Computing purity via tableaus
            elif isinstance(meas, PurityMP):
                tableau = tableau_simulator.current_inverse_tableau() ** -1
                zs = qml.math.array([tableau.z_output(k) for k in range(len(circuit.wires))])
                res = (
                    qml.math.array(1.0)
                    if circuit.op_wires == meas.wires
                    else self._measure_purity(zs, list(meas.wires))
                )

            # Computing more measurements
            else:
                raise NotImplementedError(
                    f"default.clifford doesn't support the {type(meas)} measurement at the moment."
                )

            results.append(res)

        return results

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
    def _measure_density_matrix(tableau_simulator, wires):
        """Measure the density matrix from the state of simulator device."""
        state_vector = qml.math.array(tableau_simulator.state_vector(endian="big"))
        return qml.math.reduce_dm(qml.math.einsum("i, j->ij", state_vector, state_vector), wires)

    # pylint:disable = too-many-return-statements
    def _measure_expectation(self, tableau_simulator, meas_op, stim):
        """Measure the expectation value with respect to the state of simulator device."""
        # Get the observable for the expectation value measurement
        meas_obs = meas_op.obs

        # Case for simple Pauli terms
        if isinstance(meas_obs, (qml.PauliZ, qml.PauliX, qml.PauliY)):
            pauli = stim.PauliString(_GATE_OPERATIONS[meas_obs.name])
            return qml.math.array(tableau_simulator.peek_observable_expectation(pauli))

        # Case for simple Pauli tensor
        if isinstance(meas_obs, qml.operation.Tensor):
            expec = "".join([_GATE_OPERATIONS[name] for name in meas_obs.name])
            pauli = stim.PauliString(expec)
            return qml.math.array(tableau_simulator.peek_observable_expectation(pauli))

        # Case for a Hamiltonian
        if isinstance(meas_obs, qml.Hamiltonian):
            coeffs, obs = meas_obs.terms()
            expecs = qml.math.zeros_like(coeffs)
            for idx, ob in enumerate(obs):
                expec = "".join([_GATE_OPERATIONS[name] for name in ob.name])
                pauli = stim.PauliString(expec)
                expecs[idx] = tableau_simulator.peek_observable_expectation(pauli)
            return qml.math.dot(coeffs, expecs)

        # Case for Sum
        if isinstance(meas_obs, Sum):
            return qml.math.sum(
                [
                    self._measure_expectation(tableau_simulator, ExpectationMP(op), stim)
                    for op in meas_obs
                ]
            )

        # Case for higher arithmetic depth for prod-type observables
        if meas_obs.arithmetic_depth > 1 and isinstance(meas_obs, (Prod, SProd)):
            if qml.operation.active_new_opmath():
                return self._measure_expectation(
                    tableau_simulator, ExpectationMP(meas_obs.simplify()), stim
                )
            qml.operation.enable_new_opmath()
            expec_res = self._measure_expectation(
                tableau_simulator, ExpectationMP(meas_obs.simplify()), stim
            )
            qml.operation.disable_new_opmath()
            return expec_res

        # Case for Prod
        if isinstance(meas_obs, Prod):
            expec = "".join([_GATE_OPERATIONS[op.name] for op in meas_obs.terms()[1][0]])
            pauli = stim.PauliString(expec)
            return qml.math.array(tableau_simulator.peek_observable_expectation(pauli))

        # Case for SProd
        if isinstance(meas_obs, SProd):
            coeffs, ops = meas_obs.terms()
            expec = "".join([_GATE_OPERATIONS[name] for name in ops])
            pauli = stim.PauliString(expec)
            expecs = qml.math.array(tableau_simulator.peek_observable_expectation(pauli))
            return qml.math.dot(coeffs, expecs)

        # Add support for more case when the time is right
        raise NotImplementedError(
            f"default.clifford doesn't support expectation value calculation with {type(meas_obs)} at the moment."
        )

    def _measure_variance(self, tableau_simulator, meas_op, stim):
        """Measure the variance with respect to the state of simulator device."""
        # TODO: find a better pythonic way to deal with this enable-disable
        check_op_math = qml.operation.active_new_opmath()
        qml.operation.enable_new_opmath()
        meas_obs1 = meas_op.obs.simplify()
        meas_obs2 = (meas_obs1**2).simplify()
        if not check_op_math:
            qml.operation.disable_new_opmath()

        # use the naive formula for variance, i.e., Var(Q) = ⟨𝑄^2⟩−⟨𝑄⟩^2
        return (
            self._measure_expectation(tableau_simulator, ExpectationMP(meas_obs2), stim)
            - self._measure_expectation(tableau_simulator, ExpectationMP(meas_obs1), stim) ** 2
        )

    # pylint: disable=protected-access
    @staticmethod
    def _measure_stabilizer_vn_entropy(stabilizer, wires, log_base=None):
        r"""Computes the entanglement entropy using stabilizer information.

        Computes the entanglement entropy :math:`S_A` for a subsytem described by :math:`A`,
        :math:`S_A = \text{rank}(\text{projection}_A {\mathcal{S}}) - |\mathcal{S}|`, where
        :math:`\mathcal{S}` is the stabilizer group for the system using the theory described
        in `arXiv:1901.08092 <https://arxiv.org/abs/1901.08092>`_.

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
        r"""Measure the purity of the state of simulator device

        Computes the state purity using the monotonically decreasing second-order Rényi entropy
        form given in `Sci Rep 13, 4601 (2023) <https://www.nature.com/articles/s41598-023-31273-9>`_.
        We utilize the fact that Rényi entropies are independent of the Rényi indices ``n`` for the
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

    def _measure_probability(self, tableau_simulator, circuit, meas_op, stim):
        """Measure the probability of each computational basis state."""
        if not self._tableau:
            state = self._measure_state(tableau_simulator, circuit, 0.0)
            return meas_op.process_state(state, wire_order=circuit.wires)

        if self._prob_state is None:
            raise ValueError(
                "In order to maintain computational efficiency,\
                with ``tableau=True``, the clifford device supports returning\
                probability only for selected target computational basis states.\
                Please use `probability_target()` method to set them."
            )
        assert self._prob_states.shape[1] == len(meas_op.wires)

        # we might be able to skip the inverse done below
        # as the distribution for the inverted tableau should remain same
        diagonalizing_cit = tableau_simulator.current_inverse_tableau().inverse().to_circuit()
        diagonalizing_ops = meas_op.obs.diagonalizing_gates()
        for ops in diagonalizing_ops:
            # Check if it is Clifford
            if ops not in _GATE_OPERATIONS:
                raise ValueError(
                    f"Currently, we only support observables whose diagonalizing gates are Clifford, got {ops}"
                )
            # Add to the circuit to rotate the basis
            stim_op = self.pl_to_stim(ops)
            if stim_op is not None:
                diagonalizing_cit.append(ops)

        # Iterate over the instructions and perform post-selection to obtain possible solutions
        num_states = self._prob_states.shape[0]
        prob_res = qml.math.ones(num_states)
        for ind in range(num_states):
            diagonalizing_sim = stim.TableauSimulator().do_circuit(diagonalizing_cit)
            for idx, wire in enumerate(meas_op.wires):
                outcome = self._prob_state[idx]
                expectaction = diagonalizing_sim.peek_z(wire)
                if not expectaction:
                    prob_res[ind] /= 2.0
                elif expectaction != (-1 if outcome else 1):
                    prob_res[ind] = 0.0
                    break
                diagonalizing_sim.postselect_z(wire, desired_value=outcome)

        return prob_res

    @staticmethod
    def _measure_single_sample(stim_circuit, meas_obs, stim):
        """Single sample output from a stim circuit"""
        stim_ct = stim_circuit.copy()
        stim_ct.append(f"{meas_obs[0]} {meas_obs[1]}")
        return int(stim.TableauSimulator().do_circuit(stim_ct).current_measurement_record()[0])

    def _measure_classical_shadow(self, stim_circuit, circuit, meas_op, stim):
        """Measures classical shadows from the state of simulator device"""
        meas_dict = {0: "MX", 1: "MY", 2: "MZ"}
        meas_seed = meas_op.seed or np.random.randint(2**30)

        bits = []
        recipes = np.random.RandomState(meas_seed).randint(3, size=(circuit.shots, circuit.wires))
        for recipe in recipes:
            bits.apppend(
                [
                    self._measure_single_sample(stim_circuit, [meas_dict[rec], idx], stim)
                    for idx, rec in enumerate(recipe)
                ]
            )

        return np.asarray(bits, dtype=int), np.asarray(recipes, dtype=int)

    def _measure_expval_shadow(self, stim_circuit, circuit, meas_op, stim):
        """Measures expectation value of a Pauli observable using classical shadows
        from the state of simulator device"""
        bits, recipes = self._measure_classical_shadow(stim_circuit, circuit, meas_op, stim)
        shadow = qml.shadows.ClassicalShadow(bits, recipes, wire_map=circuit.wires.tolist())
        return shadow.expval(meas_op.H, meas_op.k)
