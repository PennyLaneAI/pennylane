.. role:: html(raw)
   :format: html

Writing a new device contains three main parts:

1) Code for executing a single tape or a batches of tapes
2) Code for preprocessing and validating arbitrary input circuits
3) Binding the components together to match the device interface


A Basic Python Simulation
-------------------------

.. code-block:: python

    def sample_state(state: np.ndarray, shots: int, seed=None):
        """Generate samples from the provided state and number of shots."""
        probs = np.imag(state) ** 2 + np.real(state) ** 2
        basis_states = np.arange(len(probs))

        num_wires = int(np.log2(len(probs)))

        rng = np.random.default_rng(seed)
        basis_samples = rng.choice(basis_states, shots, p=probs)

        # convert basis state integers to array of booleans
        bin_strings = (format(s, f"0{num_wires}b") for s in basis_samples)
        return np.array([[int(val) for val in s] for s in bin_strings])


    def simulate(tape: qml.tape.QuantumTape, seed=None) -> qml.typing.Result:
        """Simulate a tape and turn it into results.

        Args:
            tape (.QuantumTape): a representation of a circuit
            seed (Any): A seed to use to control the generation of samples.

        """
        # 1) create the initial state
        state = np.zeros(2 ** len(tape.wires))
        state[0] = 1.0

        # 2) apply all the operations
        for op in tape.operations:
            op_mat = op.matrix(wire_order=tape.wires)
            state = op_mat @ state

        # 3) perform measurements
        # note that shots are pulled from the tape, not from the device
        if not tape.shots:
            results = tuple(mp.process_state(state, tape.wires) for mp in tape.measurements)
            return results[0] if len(tape.measurements) == 1 else results

        results = []
        samples = sample_state(state, shots=tape.shots.total_shots, seed=seed)
        for lower, upper in tape.shots.bins():
            sliced_samples = samples[lower: upper]
            sliced_results = tuple(mp.process_samples(sliced_samples tape.wires) for mp in tape.measurements)
            results.append(sliced_results)
        return tuple(results) if tape.shots.has_partitioned_shots else results[0]



Creating supported circuits
---------------------------

.. code-block:: python

    from pennylane.devices.preprocess import validate_device_wires, decompose, validate_measurements

    operations = frozenset({"PauliX", "PauliY", "PauliZ", "Hadamard", "CNOT", "CZ", "RX", "RY", "RZ"})

    def supports_operation(op: qml.operation.Operator) -> bool:
        """This function used by preprocessing determines what operations
        are natively supported by the device.

        While in theory ``simulate`` can support any operation with a matrix, we limit the target
        gateset for improved testing and reference purposes.

        """
        return getattr(op, "name", None) in operations

    def create_transform_program(device_wires):
        program = qml.transforms.core.TransformProgram()
        program.add_transform(validate_device_wires, wires=device_wires, name="mini.qubit")
        program.add_transform(qml.defer_measurements)
        program.add_transform(
            decompose,
            stopping_condition=supports_operation,
            skip_initial_state_prep=False,
            name="mini.qubit",
        )
        program.add_transform(qml.transforms.split_non_commuting)
        # TODO: implement a diagoanlize_measurements transform and add it here.
        program.add_transform(validate_measurements, name="mini.qubit")
        program.add_transform(qml.transforms.broadcast_expand)
        return program

Tying it all together
---------------------

Now that we have source code for both preprocessing our circuit and performing the simulation, we can
define the device itself via the ``__init__``, ``preprocess``, and ``execute`` methods.

.. code-block:: python

    from dataclasses import replace

    from pennylane.devices import Device, DefaultExecutionConfig

    @qml.devices.modifiers.simulator_tracking
    @qml.devices.modifiers.single_tape_support
    class MiniQubit(Device):

        name = "mini.qubit"

        def __init__(self, wires=None, shots=None, seed=None):
            super().__init__(wires=wires, shots=shots)

            # seed and rng not necessary for a device, but part of recommended
            # numpy practices to use a local random number generator
            self._rng = np.random.default_rng(seed)

        def preprocess(self, execution_config=DefaultExecutionConfig):
            program = create_transform_program(device_wires=self.wires)

            if "rng" in execution_config.device_options:
                return program, execution_config
            new_device_options = {"rng": self._rng}
            return program, replace(execution_config, device_options=new_device_options)

        def execute(self, circuits, execution_config=DefaultExecutionConfig):
            rng = execution_config.device_options.get("rng", self._rng)
            return tuple(simulate(tape, seed=rng) for tape in circuits)

Now we have a fully functioning device!

# TODO! This will be incorrect untill we add a diagonalize measurements transform

>>> dev = MiniQubit()
>>> @qml.qnode(dev)
... def circuit(x):
...     qml.X(0)
...     qml.IsingXX(x, wires=(0,1))
...     return qml.math.hstack([qml.expval(qml.Z(0)), qml.expval(qml.Y(0))])
>>> with dev.tracker:
...     jac = qml.jacobian(circuit)(qml.numpy.array(0.5))
>>> jac
WRONG ANSWER TILL CAN DIAGONALLIZE MEASUREMENTS
>>> dev.tracker.totals
{'batches': 2,
 'simulations': 6,
 'executions': 6,
 'results': -1.7551651237807453}