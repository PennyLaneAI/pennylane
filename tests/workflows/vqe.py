import pennylane as qml
from pennylane import numpy as np
import numpy as _np


def main(bucket_info=None, device_arn=None, display=False):

    symbols = ["H", "H"]
    coordinates = np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])
    H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)

    # assert here the H and qubits are what we expect ######
    coeffs = [-0.24274501727498227,
              -0.24274501727498227,
              -0.042072543031530185,
              0.17771358191549919,
              0.1777135819154993,
              0.12293330460167415,
              0.12293330460167415,
              0.16768338881432718,
              0.16768338881432718,
              0.1705975924056083,
              0.17627661476093914,
              -0.04475008421265302,
              -0.04475008421265302,
              0.04475008421265302,
              0.04475008421265302]

    obs = [qml.PauliZ(2),
           qml.PauliZ(3),
           qml.Identity(0),
           qml.PauliZ(0),
           qml.PauliZ(1),
           qml.PauliZ(0) @ qml.PauliZ(2),
           qml.PauliZ(1) @ qml.PauliZ(3),
           qml.PauliZ(0) @ qml.PauliZ(3),
           qml.PauliZ(1) @ qml.PauliZ(2),
           qml.PauliZ(0) @ qml.PauliZ(1),
           qml.PauliZ(2) @ qml.PauliZ(3),
           qml.PauliY(0) @ qml.PauliY(1) @ qml.PauliX(2) @ qml.PauliX(3),
           qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliY(2) @ qml.PauliY(3),
           qml.PauliY(0) @ qml.PauliX(1) @ qml.PauliX(2) @ qml.PauliY(3),
           qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliY(2) @ qml.PauliX(3)]

    manually_constructed_hamiltonian = qml.Hamiltonian(coeffs, obs)

    assert(qubits == 4)
    assert(H.compare(manually_constructed_hamiltonian))

    if display:
        print(f"Qubits: {qubits}\n"
              f"Hamiltonian: {H}")
    # end assertion checks #################################

    if bucket_info is None or device_arn is None:
        dev = qml.device(
            "default.qubit",
            wires=qubits
        )
    elif device_arn == "local_sim":
        dev = qml.device("braket.local.qubit", wires=qubits)

    else:
        dev = qml.device(
            "braket.aws.qubit",
            device_arn=device_arn,
            wires=qubits,
            s3_destination_folder=s3_bucket,
            parallel=True
        )

    electrons = 2
    hf = qml.qchem.hf_state(electrons, qubits)

    # assert here the hf is correct ########################
    manually_constructed_hf = np.array([1, 1, 0, 0])
    assert((manually_constructed_hf == hf).all())

    if display:
        print(f"Hf state: {hf}")
    # end assertion checks #################################

    def circuit(param, wires):
        qml.BasisState(hf, wires=wires)
        qml.DoubleExcitation(param, wires=[0, 1, 2, 3])

    @qml.qnode(dev)
    def cost_fn(param):
        circuit(param, wires=range(qubits))
        return qml.expval(H)

    # assert here that we get the correct expval ###########
    init_param = 0
    cost_val = cost_fn(init_param)
    computed_cost_val = -1.1173489210779484

    assert(_np.isclose(computed_cost_val, cost_val))
    # end assertion checks #################################

    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    theta = np.array(0.0, requires_grad=True)
    energy = [cost_fn(theta)]
    angle = [theta]

    max_iterations = 100
    conv_tol = 1e-06

    for n in range(max_iterations):
        theta, prev_energy = opt.step_and_cost(cost_fn, theta)

        energy.append(cost_fn(theta))
        angle.append(theta)

        conv = np.abs(energy[-1] - prev_energy)

        if display and (n % 2 == 0):
            print(f"Step = {n},  Energy = {energy[-1]:.8f} Ha")

        if conv <= conv_tol:
            break

    # assert that we converge within delta of true value ###
    delta = 1e-5
    true_energy = -1.136189454088
    predicted_energy = energy[-1]

    assert(_np.isclose(true_energy, predicted_energy, rtol=delta, atol=delta))

    if display:
        print(f"Completed workflow with: Energy = {energy[-1]} Ha, Angle = {angle[-1]} Rad")
    # end assertion checks #################################

    return


if __name__ == "__main__":
    my_bucket = "amazon-braket-Bucket-Name"
    my_prefix = "Folder-Name"
    s3_bucket = (my_bucket, my_prefix)

    state_vector_sim_device_arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
    local_sim = "local_sim"

    # main(display=True)
    # main(s3_bucket, state_vector_sim_device_arn, display=True)
    main(None, local_sim, display=False)
