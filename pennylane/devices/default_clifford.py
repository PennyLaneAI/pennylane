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
This module contains the next generation successor to default qubit
"""

from dataclasses import replace
from functools import partial
from numbers import Number
from typing import Union, Callable, Tuple, Optional, Sequence
import concurrent.futures
import numpy as np

import pennylane as qml
from pennylane.tape import QuantumTape, QuantumScript
from pennylane.typing import Result, ResultBatch
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.transforms.core import TransformProgram
from pennylane.measurements.expval import ExpectationMP
from pennylane.devices.qubit.sampling import get_num_shots_and_executions
from pennylane.devices.qubit.simulate import INTERFACE_TO_LIKE

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
QuantumTapeBatch = Sequence[QuantumTape]
QuantumTape_or_Batch = Union[QuantumTape, QuantumTapeBatch]
# always a function from a resultbatch to either a result or a result batch
PostprocessingFn = Callable[[ResultBatch], Result_or_ResultBatch]

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
    "Snapshot": None,
    "BasisState": None,
    "StatePrep": None,
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
}


def observable_stopping_condition(obs: qml.operation.Operator) -> bool:
    """Specifies whether or not an observable is accepted by DefaultClifford."""
    return obs.name in _MEAS_OBSERVABLES


def operations_stopping_condition(obs: qml.operation.Operator) -> bool:
    """Specifies whether or not an observable is accepted by DefaultClifford."""
    return obs.name in _GATE_OPERATIONS


def _import_stim():
    """Import stim."""
    try:
        # pylint: disable=import-outside-toplevel, unused-import, multiple-imports
        import stim
    except ImportError as Error:
        raise ImportError(
            "This feature requires stim, a fast stabilizer circuit simulator."
            "It can be installed with: pip install stim."
        ) from Error
    return stim


class DefaultClifford(Device):
    """A PennyLane device written in Python and capable of executing Clifford circuit using `Stim (2021) <https://github.com/quantumlib/Stim/tree/main>`_  .

    Args:
        shots (int, Sequence[int], Sequence[Union[int, Sequence[int]]]): The default number of shots to use in executions involving
            this device.
        check_clifford (bool): Check if all the gate operations in the circuits to be executed are Clifford. Default is ``True``.
        max_error (float): The maximum permissible decomposition error for the circuits with non-Clifford gate operaitons.
            Default is ``0.0``.
        state (str): Describes what should be returned when the device's state is computed with ``qml.state``. Default is
            "tableau", which makes it return the final evolved Tableau. Alternatively, one may use "state_vector" to obtain
            the evolved state vector. Note that the latter might not be computationally feasible for larger qubit numbers.
        seed (Union[str, None, int, array_like[int], SeedSequence, BitGenerator, Generator, jax.random.PRNGKey]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``, or
            a request to seed from numpy's global random number generator.
            The default, ``seed="global"`` pulls a seed from NumPy's global generator. ``seed=None``
            will pull a seed from the OS entropy.
            If a ``jax.random.PRNGKey`` is passed as the seed, a JAX-specific sampling function using
            ``jax.random.choice`` and the ``PRNGKey`` will be used for sampling rather than
            ``numpy.random.default_rng``.
        max_workers (int): A ``ProcessPoolExecutor`` executes tapes asynchronously
            using a pool of at most ``max_workers`` processes. If ``max_workers`` is ``None``,
            only the current process executes tapes. If you experience any
            issue, say using JAX, TensorFlow, Torch, try setting ``max_workers`` to ``None``.

    **Example:**

    .. code-block:: python

        n_layers = 5
        n_wires = 10
        num_qscripts = 5

        shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires)
        rng = qml.numpy.random.default_rng(seed=42)

        qscripts = []
        for i in range(num_qscripts):
            params = rng.random(shape)
            op = qml.StronglyEntanglingLayers(params, wires=range(n_wires))
            qs = qml.tape.QuantumScript([op], [qml.expval(qml.PauliZ(0))])
            qscripts.append(qs)

    >>> dev = DefaultClifford()
    >>> program, execution_config = dev.preprocess()
    >>> new_batch, post_processing_fn = program(qscripts)
    >>> results = dev.execute(new_batch, execution_config=execution_config)
    >>> post_processing_fn(results)
    [-0.0006888975950537501,
    0.025576307134457577,
    -0.0038567269892757494,
    0.1339705146860149,
    -0.03780669772690448]

    .. details::
        :title: Tracking

        ``DefaultClifford`` tracks:

        * ``executions``: the number of unique circuits that would be required on quantum hardware
        * ``shots``: the number of shots
        * ``resources``: the :class:`~.resource.Resources` for the executed circuit.
        * ``simulations``: the number of simulations performed. One simulation can cover multiple QPU executions, such as for non-commuting measurements and batched parameters.
        * ``batches``: The number of times :meth:`~.execute` is called.
        * ``results``: The results of each call of :meth:`~.execute`
        * ``derivative_batches``: How many times :meth:`~.compute_derivatives` is called.
        * ``execute_and_derivative_batches``: How many times :meth:`~.execute_and_compute_derivatives` is called
        * ``vjp_batches``: How many times :meth:`~.compute_vjp` is called
        * ``execute_and_vjp_batches``: How many times :meth:`~.execute_and_compute_vjp` is called
        * ``jvp_batches``: How many times :meth:`~.compute_jvp` is called
        * ``execute_and_jvp_batches``: How many times :meth:`~.execute_and_compute_jvp` is called
        * ``derivatives``: How many circuits are submitted to :meth:`~.compute_derivatives` or :meth:`~.execute_and_compute_derivatives`.
        * ``vjps``: How many circuits are submitted to :meth:`~.compute_vjp` or :meth:`~.execute_and_compute_vjp`
        * ``jvps``: How many circuits are submitted to :meth:`~.compute_jvp` or :meth:`~.execute_and_compute_jvp`


    .. details::
        :title: Accelerate calculations with multiprocessing

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

        * OMP_NUM_THREADS
        * MKL_NUM_THREADS
        * OPENBLAS_NUM_THREADS

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
        max_error=0.0,
        state="tableau",
        seed="global",
        max_workers=None,
    ) -> None:
        super().__init__(wires=wires, shots=shots)

        self._stim = _import_stim()
        self._max_workers = max_workers
        self._check_clifford = check_clifford
        self._max_error = max(max_error, 0.0)

        if state in ["tableau", "state_vector"]:
            self._state = state
        else:
            raise ValueError(
                f"Keyword state only accepts two options: 'tableau' and 'state_vector', got {state}."
            )

        seed = np.random.randint(0, high=10000000) if seed == "global" else seed
        if qml.math.get_interface(seed) == "jax":
            self._prng_key = seed
            self._rng = np.random.default_rng(None)
        else:
            self._prng_key = None
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
        if execution_config.gradient_method == "best":
            updated_values["gradient_method"] = None
        if execution_config.use_device_gradient is None:
            updated_values["use_device_gradient"] = True
        if execution_config.use_device_jacobian_product is not None:
            updated_values["use_device_jacobian_product"] = None
        if execution_config.grad_on_execution is None:
            updated_values["grad_on_execution"] = execution_config.gradient_method == "adjoint"
        updated_values["device_options"] = dict(execution_config.device_options)  # copy
        if "max_workers" not in updated_values["device_options"]:
            updated_values["device_options"]["max_workers"] = self._max_workers
        if "rng" not in updated_values["device_options"]:
            updated_values["device_options"]["rng"] = self._rng
        if "prng_key" not in updated_values["device_options"]:
            updated_values["device_options"]["prng_key"] = self._prng_key
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

        * Supports any qubit operations that provide a matrix
        * Currently does not support finite shots
        * Currently does not intrinsically support parameter broadcasting

        """
        config = self._setup_execution_config(execution_config)
        transform_program = TransformProgram()

        transform_program.add_transform(validate_device_wires, self.wires, name=self.name)
        transform_program.add_transform(qml.defer_measurements, device=self)

        # TODO: Add the Clifford+T decomposition transform here instead.
        if self._check_clifford:
            transform_program.add_transform(
                decompose, stopping_condition=operations_stopping_condition, name=self.name
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
        interface = (
            execution_config.interface
            if execution_config.gradient_method in {"backprop", None}
            else None
        )
        if max_workers is None:
            results = tuple(
                self.simulate(
                    c,
                    rng=self._rng,
                    prng_key=self._prng_key,
                    debugger=self._debugger,
                    interface=interface,
                )
                for c in circuits
            )
        else:
            vanilla_circuits = [convert_to_numpy_parameters(c) for c in circuits]
            seeds = self._rng.integers(2**31 - 1, size=len(vanilla_circuits))
            _wrap_simulate = partial(self.simulate, debugger=None, interface=interface)
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                exec_map = executor.map(
                    _wrap_simulate,
                    vanilla_circuits,
                    seeds,
                    [self._prng_key] * len(vanilla_circuits),
                )
                results = tuple(exec_map)

            # reset _rng to mimic serial behavior
            self._rng = np.random.default_rng(self._rng.integers(2**31 - 1))

        if self.tracker.active:
            self.tracker.update(batches=1)
            self.tracker.record()
            for i, c in enumerate(circuits):
                qpu_executions, shots = get_num_shots_and_executions(c)
                res = np.array(results[i]) if isinstance(results[i], Number) else results[i]
                if c.shots:
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

        res = tuple(0.0 for circuit in circuits)

        max_workers = execution_config.device_options.get("max_workers", self._max_workers)
        if max_workers is not None:
            # reset _rng to mimic serial behavior
            self._rng = np.random.default_rng(self._rng.integers(2**31 - 1))

        return res[0] if is_single_circuit else res

    def execute_and_compute_derivatives(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        is_single_circuit = False
        if isinstance(circuits, QuantumScript):
            is_single_circuit = True
            circuits = [circuits]

        if self.tracker.active:
            for c in circuits:
                self.tracker.update(resources=c.specs["resources"])
            self.tracker.update(
                execute_and_derivative_batches=1,
                executions=len(circuits),
                derivatives=len(circuits),
            )
            self.tracker.record()

        meas = self.execute(circuits, execution_config=execution_config)
        grad = self.compute_derivatives(circuits, execution_config=execution_config)

        results, jacs = tuple(zip(meas, grad))

        return (results[0], jacs[0]) if is_single_circuit else (results, jacs)

    def supports_jvp(
        self,
        execution_config: Optional[ExecutionConfig] = None,
        circuit: Optional[QuantumTape] = None,
    ) -> bool:
        """Whether or not this device defines a custom jacobian vector product.

        ``DefaultClifford`` returns trivial derivates everytime.

        Args:
            execution_config (ExecutionConfig): The configuration of the desired derivative calculation
            circuit (QuantumTape): An optional circuit to check derivatives support for.

        Returns:
            bool: Whether or not a derivative can be calculated provided the given information
        """
        return False

    def supports_vjp(
        self,
        execution_config: Optional[ExecutionConfig] = None,
        circuit: Optional[QuantumTape] = None,
    ) -> bool:
        """Whether or not this device defines a custom vector jacobian product.

        ``DefaultClifford`` returns trivial derivates everytime.

        Args:
            execution_config (ExecutionConfig): A description of the hyperparameters for the desired computation.
            circuit (None, QuantumTape): A specific circuit to check differentation for.

        Returns:
            bool: Whether or not a derivative can be calculated provided the given information
        """
        return False

    # pylint: disable=unidiomatic-typecheck, unused-argument
    def simulate(
        self,
        circuit: qml.tape.QuantumScript,
        rng=None,
        prng_key=None,
        debugger=None,
        interface=None,
    ) -> Result:
        """Simulate a single quantum script.

        Args:
            circuit (QuantumTape): The single circuit to simulate
            rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
                seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
                If no value is provided, a default RNG will be used.
            prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
                the key to the JAX pseudo random number generator. If None, a random key will be
                generated. Only for simulation using JAX.
            debugger (_Debugger): The debugger to use
            interface (str): The machine learning interface to create the initial state with

        Returns:
            tuple(TensorLike): The results of the simulation

        Note that this function can return measurements for non-commuting observables simultaneously.

        This function assumes that all operations provide matrices.

        >>> qs = qml.tape.QuantumScript([qml.RX(1.2, wires=0)], [qml.expval(qml.PauliZ(0)), qml.probs(wires=(0,1))])
        >>> simulate(qs)
        (0.36235775447667357,
        tensor([0.68117888, 0.        , 0.31882112, 0.        ], requires_grad=True))

        """

        stim = self._stim

        circuit = circuit.map_to_standard_wires()

        if circuit.shots:
            raise NotImplementedError(
                "default.clifford currently doesn't support computation with shots."
            )

        prep = None
        if len(circuit) > 0 and isinstance(circuit[0], qml.operation.StatePrepBase):
            prep = circuit[0]

        # initial state is batched only if the state preparation (if it exists) is batched
        is_state_batched = bool(prep and prep.batch_size is not None)
        if is_state_batched:
            raise NotImplementedError("Clifford simulator doesn't support batching.")

        stim_ct = stim.Circuit()
        initial_tableau = stim.Tableau.from_circuit(stim_ct)

        tableau_simulator = stim.TableauSimulator()

        use_prep_ops = bool(prep)
        if use_prep_ops:
            initial_tableau = stim.Tableau.from_state_vector(
                qml.math.reshape(prep.state_vector(wire_order=list(circuit.op_wires)), (1, -1))[0],
                endian="big",
            )
            tableau_simulator.do_tableau(initial_tableau, circuit.wires)

        global_phase_ops = []
        for op in circuit.operations[use_prep_ops:]:
            gate, wires = _GATE_OPERATIONS[op.name], op.wires
            if gate is not None:
                stim_ct.append(gate, wires)
            else:
                if op.name == "GlobalPhase":
                    global_phase_ops.append(op)
                elif op.name == "Snapshot":
                    state = stim.Tableau.from_circuit(stim_ct).to_state_vector()
                    if debugger is not None and debugger.active:
                        flat_state = qml.math.flatten(state)
                    if op.tag:
                        debugger.snapshots[op.tag] = flat_state
                    else:
                        debugger.snapshots[len(debugger.snapshots)] = flat_state
                else:
                    pass
        tableau_simulator.do_circuit(stim_ct)

        res = []
        for meas in circuit.measurements:
            # Analytic case
            if not circuit.shots:
                # Computing statevector via tableaus
                if type(meas) is qml.measurements.StateMP:
                    if self._state == "tableau":
                        inverse_tableau = tableau_simulator.current_inverse_tableau()
                        res.append(qml.math.array(inverse_tableau.inverse().to_numpy()))
                    else:
                        state_vector = qml.math.array(
                            tableau_simulator.state_vector(endian="big"),
                            like=INTERFACE_TO_LIKE[interface],
                        )
                        res.append(state_vector)

                # Computing density matrix via tableaus
                elif type(meas) is qml.measurements.DensityMatrixMP:
                    state_vector = qml.math.array(
                        tableau_simulator.state_vector(endian="big"),
                        like=INTERFACE_TO_LIKE[interface],
                    )
                    density_matrix = qml.math.einsum("i, j->ij", state_vector, state_vector)
                    res.append(density_matrix)

                # Computing purity via tableaus // Trivial
                elif type(meas) is qml.measurements.PurityMP:
                    res.append(qml.math.array(1.0, like=INTERFACE_TO_LIKE[interface]))

                # Computing expectation values via measurement
                elif isinstance(meas, ExpectationMP):
                    # Case for simple Pauli terms
                    if (
                        isinstance(meas.obs, qml.PauliZ)
                        or isinstance(meas.obs, qml.PauliX)
                        or isinstance(meas.obs, qml.PauliY)
                    ):
                        pauli = stim.PauliString(_GATE_OPERATIONS[meas.obs.name])
                        res.append(tableau_simulator.peek_observable_expectation(pauli))

                    # Case for simple Pauli tensor
                    elif isinstance(meas.obs, qml.operation.Tensor):
                        expec = "".join([_GATE_OPERATIONS[name] for name in meas.obs.name])
                        pauli = stim.PauliString(expec)
                        res.append(tableau_simulator.peek_observable_expectation(pauli))

                    # Case for a Hamiltonian
                    elif isinstance(meas.obs, qml.Hamiltonian):
                        coeffs, obs = meas.obs.terms()
                        expecs = qml.math.zeros_like(coeffs)
                        for idx, ob in enumerate(obs):
                            expec = "".join([_GATE_OPERATIONS[name] for name in ob.name])
                            pauli = stim.PauliString(expec)
                            expecs[idx] = tableau_simulator.peek_observable_expectation(pauli)
                        res.append(qml.math.sum(coeffs * expecs))

                    # Add support for more case when the time is right
                    else:
                        raise NotImplementedError(
                            f"default.clifford doesn't support {meas} at the moment."
                        )

        return tuple(res)
