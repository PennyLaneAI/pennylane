#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pennylane as qml
from pennylane.ops.decompositions import DecompositionLibrary, DecompositionGraph


# In[2]:


library = DecompositionLibrary()


def crx_to_rx_cz(*params, wires=None):
    return [
        qml.RX(params[0] / 2, wires=wires[1]),
        qml.CZ(wires=wires),
        qml.RX(-params[0] / 2, wires=wires[1]),
        qml.CZ(wires=wires),
    ]


library.register_static_decomposition(qml.CRX, crx_to_rx_cz)


def crx_to_rz_ry(*params, wires=None):
    return [
        qml.RZ(np.pi / 2, wires=wires[1]),
        qml.RY(params[0] / 2, wires=wires[1]),
        qml.CNOT(wires=wires),
        qml.RY(-params[0] / 2, wires=wires[1]),
        qml.CNOT(wires=wires),
        qml.RZ(-np.pi / 2, wires=wires[1]),
    ]


library.register_static_decomposition(qml.CRX, crx_to_rz_ry)


def crx_to_h_crz(*params, wires=None):
    return [
        qml.Hadamard(wires=wires[1]),
        qml.CRZ(params[0], wires=wires),
        qml.Hadamard(wires=wires[1]),
    ]


library.register_static_decomposition(qml.CRX, crx_to_h_crz)


def crz_to_rz_cnot(*params, wires=None):
    return [
        qml.RZ(params[0] / 2, wires=wires[1]),
        qml.CNOT(wires=wires),
        qml.RZ(-params[0] / 2, wires=wires[1]),
        qml.CNOT(wires=wires),
    ]


library.register_static_decomposition(qml.CRZ, crz_to_rz_cnot)


def hadamard_to_rx_rz(*params, wires=None):
    return [
        qml.RZ(np.pi / 2, wires=wires),
        qml.RX(np.pi / 2, wires=wires),
        qml.RZ(np.pi / 2, wires=wires),
    ]


library.register_static_decomposition(qml.Hadamard, hadamard_to_rx_rz)


def cnot_to_cz_h(*params, wires=None):
    return [
        qml.Hadamard(wires=wires[1]),
        qml.CZ(wires=wires),
        qml.Hadamard(wires=wires[1]),
    ]


library.register_static_decomposition(qml.CNOT, cnot_to_cz_h)


def cz_to_cnot_h(*params, wires=None):
    return [
        qml.Hadamard(wires=wires[1]),
        qml.CNOT(wires=wires),
        qml.Hadamard(wires=wires[1]),
    ]


library.register_static_decomposition(qml.CZ, cz_to_cnot_h)


# In[6]:


circuit_operations = [qml.CRX(0.5, wires=[0, 1])]
supported_operations = {"RZ", "RX", "CZ", "Hadamard"}

graph = DecompositionGraph(circuit_operations, supported_operations, library)
graph.solve(lazy=False)
graph.decompose(circuit_operations[0])
