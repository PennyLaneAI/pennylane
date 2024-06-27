
def simulate(
    circuit: qml.tape.QuantumScript, rng=None, prng_key=None, debugger=None, interface=None
) -> Result:
    """TODO

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

    >>> qs = qml.tape.QuantumScript([qml.TRX(1.2, wires=0)], [qml.expval(qml.GellMann(0, 3)), qml.probs(wires=(0,1))])
    >>> simulate(qs)
    (0.36235775447667357,
    tensor([0.68117888, 0.        , 0.        , 0.31882112, 0.        , 0.        ], requires_grad=True))

    """
    state, is_state_batched = get_final_state(
        circuit, debugger=debugger, interface=interface, rng=rng, prng_key=prng_key
    )
    return measure_final_state(circuit, state, is_state_batched, rng=rng, prng_key=prng_key)


def get_final_state(circuit, debugger=None, interface=None, **kwargs):
    """
    TODO

    Args:
        circuit (.QuantumScript): The single circuit to simulate
        debugger (._Debugger): The debugger to use
        interface (str): The machine learning interface to create the initial state with

    Returns:
        Tuple[TensorLike, bool]: A tuple containing the final state of the quantum script and
            whether the state has a batch dimension.

    """
    circuit = circuit.map_to_standard_wires()
    state = create_initial_state(sorted(circuit.op_wires), like="jax")

    op_array = []
    for op in circuit.operations:

        # op_index = None
        # for i, op_type in enumerate(op_types):
        #     if isinstance(op, op_type):
        #         op_index = i
        # if op_index is None:
        #     raise ValueError("This simulator only supports")

        op_index = op_types.index(type(op))

        wires = op.wires()
        if len(wires) == 1:
            wires = [-1, wires[0]]
        if len(wires) == 2:
            wires = list(wires)
        params = op.parameters + ([0] * (3-op.num_params))
        op_array.append([[op_index] + wires, params])

    return state


def get_final_state(circuit, debugger=None, interface=None, **kwargs):
    """
    TODO

    Args:
        circuit (.QuantumScript): The single circuit to simulate
        debugger (._Debugger): The debugger to use
        interface (str): The machine learning interface to create the initial state with

    Returns:
        Tuple[TensorLike, bool]: A tuple containing the final state of the quantum script and
            whether the state has a batch dimension.

    """
    circuit = circuit.map_to_standard_wires()

    prep = None
    if len(circuit) > 0 and isinstance(circuit[0], qml.operation.StatePrepBase):
        prep = circuit[0]

    state = create_initial_state(sorted(circuit.op_wires), prep, like=INTERFACE_TO_LIKE[interface])

    # initial state is batched only if the state preparation (if it exists) is batched
    is_state_batched = bool(prep and prep.batch_size is not None)
    for op in circuit.operations[bool(prep) :]:
        state = apply_operation(
            op,
            state,
            is_state_batched=is_state_batched,
            debugger=debugger,
            tape_shots=circuit.shots,
            **kwargs,
        )

        # new state is batched if i) the old state is batched, or ii) the new op adds a batch dim
        is_state_batched = is_state_batched or op.batch_size is not None

    num_operated_wires = len(circuit.op_wires)
    for i in range(len(circuit.wires) - num_operated_wires):
        # If any measured wires are not operated on, we pad the density matrix with zeros.
        # We know they belong at the end because the circuit is in standard wire-order
        # Since it is a dm, we must pad it with 0s on the last row and last column
        current_axis = num_operated_wires + i + is_state_batched
        state = qml.math.stack(
            ([state] + [qml.math.zeros_like(state)] * (QUDIT_DIM - 1)), axis=current_axis
        )
        state = qml.math.stack(([state] + [qml.math.zeros_like(state)] * (QUDIT_DIM - 1)), axis=-1)

    return state, is_state_batched