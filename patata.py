import pennylane as qml
import numpy as np



print(qml.math.shape([1,2,3])[0])
"""
poly = [0, 1.0, 0, -1 / 2, 0, 1 / 3, 0]
angles = qml.poly_to_angles(poly, "QSVT")

poly_tf = tf.Variable(poly)

angles_tf = qml.poly_to_angles(poly_tf, "QSVT")

print(qml.math.allclose(angles, angles_tf))
"""
