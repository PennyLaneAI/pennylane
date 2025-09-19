import numpy as np

import pennylane as qml

from pennylane.ops import X, Y, Z, H
from pennylane.devices.preprocess import measurements_from_samples
from pennylane.ftqc import RotXZX, convert_to_mbqc_formalism, diagonalize_mcms

from functools import partial

qml.decomposition.enable_graph()

dev = qml.device("lightning.qubit", wires=15)

# sign flip on Y(0) and Y(1)
# def ops():
#     RotXZX(0.64, 0.33, 1.2, 0)
#     RotXZX(0.45, 0.13, 0.78, 1)
#     H(0)
#     Z(0)

# sign flip on X(0) and Y(1)
def ops2():
    RotXZX(0.64, 0.33, 1.2, 0)
    RotXZX(np.pi/4, 0.54, 0, 1)
    H(0)
    Z(0)
    Y(1)
    Z(1)
    X(0)
    H(1)

ops = ops2

@qml.qnode(dev)
def analytic_circ(obs):
    ops()
    return qml.expval(obs(0)), qml.expval(obs(1))


@measurements_from_samples
@partial(qml.transforms.diagonalize_measurements, use_op_gates_only=True)
@qml.transforms.split_non_commuting
@qml.set_shots(5000)
@qml.qnode(dev, mcm_method="one-shot")
def circuit_mbqc(obs):
    ops()
    return qml.expval(obs(0)), qml.expval(obs(1))


analytic_circ(X), analytic_circ(Y), analytic_circ(Z)

from functools import partial
from pennylane.ftqc.pauli_tracker import apply_byproduct_corrections


def get_mbqc_result(obs, use_pauli_tracker=True):

    # get the mbqc gate-set tape
    initial_tape = qml.workflow.construct_tape(circuit_mbqc, level="user")(obs)

    # convert to the mbqc formalism and diagonalize
    (tape,), _ = convert_to_mbqc_formalism(initial_tape, diagonalize_mcms=True, pauli_tracker=use_pauli_tracker)

    # apply default device preprocessing (includes dynamic-one-shot transform)
    program, config = dev.preprocess()
    tapes, _ = program([tape])

    res = dev.execute(tapes, execution_config=config)
    observable_measurements = [r[0][0] for r in res[0]]
    final_samples = []
    if use_pauli_tracker:
        # correct the samples based on mcms to get the final samples\n",
        mid_measures = [r[1:] for r in res[0]]
        correction_fn = partial(apply_byproduct_corrections, initial_tape)
        for samples, mcms in zip(observable_measurements, mid_measures):
            final_samples.append(correction_fn(mcms, samples))
    else:
        # if we aren't using the pauli tracker, the initial measurements are the final samples\n",
        final_samples = observable_measurements
    return (np.mean([-1 if r[0] else 1 for r in final_samples]), np.mean([-1 if r[1] else 1 for r in final_samples]))


print(get_mbqc_result(X), analytic_circ(X))
print(get_mbqc_result(Y), analytic_circ(Y))
print(get_mbqc_result(Z), analytic_circ(Z))