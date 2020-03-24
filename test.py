import pennylane as qml
import tensorflow as tf
import sklearn.datasets

n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qnode(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=list(range(n_qubits)))
    qml.templates.StronglyEntanglingLayers(weights, wires=list(range(n_qubits)))
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

weight_shapes = {"weights": (3, n_qubits, 3)}

q_layer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=2)
model = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(2), q_layer, tf.keras.layers.Dense(2, activation="softmax"),]
)

data = sklearn.datasets.make_moons()
X = tf.constant(data[0])
Y = tf.one_hot(data[1], depth=2)
