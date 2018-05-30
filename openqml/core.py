# Copyright 2018 Xanadu Quantum Technologies Inc.
r"""
Core classes
============

**Module name:** :mod:`openqml.core`

.. currentmodule:: openqml.core


Classes
-------

.. autosummary::
   Optimizer
"""

#from numpy import pi, cos, sin, exp, sqrt, arctan, arccosh, sign, arctan2, arcsinh, cosh, tanh, ndarray, all, arange
from .circuit import ParRef


class Optimizer:
    """Quantum circuit optimizer.

    Args:
      pname (str): name of plugin to load
    """
    def __init__(self, pname):
        self.plugin = load_plugin(pname)  #: Plugin: backend plugin for executing quantum circuits

    def __str__(self):
        """String representation."""
        return self.__class__.__name__

    def set_circuit(self, circuit):
        self.circuit = circuit

    def eval(self, params, **kwargs):
        return self.plugin.execute_circuit(self.circuit, params, **kwargs)

    def gradient_finite_diff(self, params, h=1e-7, **kwargs):
        """Compute a circuit gradient using finite differences.

        Given an n-parameter quantum circuit, this function computes its gradient with respect to the parameters
        using the finite difference method. The current implementation evaluates the circuit at n+1 points of the parameter space.

        Args:
          circuit (Circuit): quantum circuit to differentiate
          params (Sequence[float]): point in parameter space at which to evaluate the gradient
          h (float): step size
        Returns:
          array: gradient vector
        """
        params = np.asarray(params)
        grad = np.zeros(params.shape)
        # value at the evaluation point
        x0 = self.plugin.execute_circuit(self.circuit, params, **kwargs)
        for k in range(len(params)):
            # shift the k:th parameter by h
            temp = params.copy()
            temp[k] += h
            x = self.plugin.execute_circuit(self.circuit, temp, **kwargs)
            grad[k] = (x-x0) / h
        return grad

    def gradient_single_qubit(self, params, **kwargs):
        """Compute a circuit gradient using the angle method.

        Given an n-parameter quantum circuit, this function computes its gradient with respect to the parameters
        using the angle method. The method only works for one-parameter single-qubit gates where the parameter is the rotation angle.
        The circuit is evaluated twice for each incidence of each parameter in the circuit.

        Args:
          circuit (Circuit): quantum circuit to differentiate
          params (Sequence[float]): point in parameter space at which to evaluate the gradient
        Returns:
          array: gradient vector
        """
        params = np.asarray(params)
        grad = np.zeros(params.shape)
        n = self.circuit.n_par
        for k in range(n):
            # find the Commands in which the parameter appears, use the product rule
            for c in self.circuit.pars[k]:
                if c.gate.n_par != 1:
                    raise ValueError('For now we can only differentiate one-parameter gates.')
                # we temporarily edit the Command so that parameter k is replaced by a new one,
                # which we can modify without affecting other Commands depending on the original.
                temp = c.par[0]
                assert(temp.idx == k)
                c.par[0] = ParRef(n)
                # shift it by pi/2 and -pi/2
                temp = np.r_[params, params[k]+np.pi/2]
                x2 = self.plugin.execute_circuit(self.circuit, temp, **kwargs)
                temp[-1] = params[k] -np.pi/2
                x1 = self.plugin.execute_circuit(self.circuit, temp, **kwargs)
                # restore the original parameter
                c.par[0] = temp
                grad[k] += (x2-x1) / 2
        return grad
