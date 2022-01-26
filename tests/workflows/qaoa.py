import pennylane as qml
from pennylane import qaoa
from pennylane import numpy as np
import networkx as nx

def main(bucket_info=None, device_arn=None, display=False):

    edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
    graph = nx.Graph(edges)


    cost_h, mixer_h = qaoa.min_vertex_cover(graph, constrained=False)

    if display:
        print("Cost Hamiltonian", cost_h)
        print("Mixer Hamiltonian", mixer_h)

    def qaoa_layer(gamma, alpha):
        qaoa.cost_layer(gamma, cost_h)
        qaoa.mixer_layer(alpha, mixer_h)

    wires = range(4)
    depth = 2

    def circuit(params, **kwargs):
        for w in wires:
            qml.Hadamard(wires=w)
        qml.layer(qaoa_layer, depth, params[0], params[1])

    if bucket_info is None or device_arn is None:
        dev = qml.device(
            "default.qubit",
            wires=wires
        )
    else:
        dev = qml.device(
            "braket.aws.qubit",
            device_arn=device_arn,
            wires=wires,
            s3_destination_folder=s3_bucket,
            parallel=True
        )

    @qml.qnode(dev)
    def cost_function(params):
        circuit(params)
        return qml.expval(cost_h)

    optimizer = qml.GradientDescentOptimizer()
    steps = 70
    params = np.array([[0.5, 0.5], [0.5, 0.5]], requires_grad=True)

    for i in range(steps):
        params = optimizer.step(cost_function, params)

    if display:
        print("Optimal Parameters")
        print(params)


    @qml.qnode(dev)
    def probability_circuit(gamma, alpha):
        circuit([gamma, alpha])
        return qml.probs(wires=wires)


    probs = probability_circuit(params[0], params[1])

    max_values = [i for i, x in enumerate(probs) if x == max(probs)]

    assert max_values == [6, 10]

    if display:
        print("probs", probs)
        print("max_values", max_values)

if __name__ == "__main__":
    my_bucket = "amazon-braket-Bucket-Name"
    my_prefix = "Folder-Name"
    s3_bucket = (my_bucket, my_prefix)

    state_vector_sim_device_arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"

    main(display=False)
    # main(s3_bucket, state_vector_sim_device_arn, display=True)