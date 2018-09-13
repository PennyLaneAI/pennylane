import autograd as ag
import autograd.numpy as np
from autograd.builtins import list
#from autograd.numpy.random import (randn,)

import openqml as qm


def circuit_nested_pars(x, y, arr):
    #qm.Rot(x, 0.3, -0.2, [0])
    qm.RX(x, [0])
    qm.RY(arr[0][1], [0])
    #qm.SWAP([0, 1])
    return qm.expectation.PauliZ(0)


def circuit(x, y, arr):
    qm.RX(x, [0])
    qm.RY(arr[1], [0])
    return qm.expectation.PauliZ(0)


def pars(a, b=np.pi/2*0.99):
    # nested list of parameters
    #return list([a, 0.2, list([np.array([666.0, b]), 777.0])])

    # flat list of parameters
    return list([b, 0.73, np.array([666.0, a])])


if __name__ == '__main__':

    dev = qm.device('default.qubit', wires=2)
    q = qm.QNode(circuit, dev)

    def cost(x):
        p = pars(x)
        temp = 1 * q(*p)
        print('returning:', temp)
        return temp

    #g_cost = ag.jacobian(cost, 0)
    g_cost = ag.grad(cost, 0)

    p = pars(0.2)
    print('qnode:', q(*p))
    print('qnode grad:', q.gradient(p))
    print('cost:', cost(0.2))
    print('cost auto grad:', g_cost(0.2))
