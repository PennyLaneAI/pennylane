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


Optimizer methods
-----------------

.. currentmodule:: openqml.core.Optimizer

.. autosummary::
    train
    weights
    reg_cost_L2

----
"""

import signal
import logging as log

import numpy as np
from scipy.optimize import minimize


# optimization parameters
#Par = namedtuple('Par', 'name, init, regul')
#Par.__new__.__defaults__ = ('', 0, False)


OPTIMIZER_NAMES = ["SGD", "Nelder-Mead", "Powell", "CG", "BFGS", "Newton-CG",
                   "L-BFGS-B", "TNC", "COBYLA", "SLSQP", "dogleg", "trust-ncg",
                   "trust-exact", "trust-krylov"]


class Optimizer:
    """Quantum circuit optimizer.

    cost_func typically involves the evaluation of one or more :class:`QNode` s
    representing variational quantum circuits.

    .. todo:: Compute cost_grad using automatic differentiation.

    Optimization hyperparameters are given as keyword arguments.

    Args:
      cost_func (callable): cost/error function
      cost_grad (callable): gradient of the cost/error function
      weights (array[float]): initial values for the weights/optimization parameters
      n_data (int): total number of data samples to be used in training

    Keyword Args:
      optimizer (str): 'SGD' or any Hessian/Jacobian-free default optimizer compatible with scipy's minimize method: "Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B", "TNC", "COBYLA", "SLSQP"
      init_learning_rate (float): initial learning rate, usually around 0.1
      decay  (float): decay rate for the learning rate
      lambda (float): regularization strength
      regularizer (None, callable):  None, 'L2', or a custom function mapping Sequence[float] to float.

    Based on MLtoolbox by Maria Schuld.
    """
    def __str__(self):
        """String representation."""
        return self.__class__.__name__

    def __init__(self, cost_func, cost_grad, weights, n_data, **kwargs):

        self._cost_func = cost_func  #: callable: scalar function to be minimized
        self._cost_grad = cost_grad  #: callable: gradient of cost_fun
        self._n_data = n_data  #: int: total number of data samples to be used in training
        self.stop = False  #: bool: flag, stop optimization

        # default hyperparameters
        default_hp = {'optimizer': 'SGD',
                      'init_learning_rate': 0.01,
                      'decay': 0.,
                      'lambda': 0.,
                      'regularizer': 'None',
        }
        self._hp = default_hp    #: dict[str->*]: hyperparameters
        self._hp.update(kwargs)  # update with user-given hyperparameters
        print("\n-----------------------------\n HYPERPARAMETERS: \n")
        for key in sorted(self._hp):
            temp = '' if key in kwargs else ' (default)'
            print('{}\t\t{}{}'.format(key, self._hp[key], temp))
        print("\n-----------------------------")

        temp = self._hp['optimizer']
        if not callable(temp) and temp not in OPTIMIZER_NAMES:
            raise ValueError("The optimizer has to be either a callable or in the list of allowed optimizers, {}".format(OPTIMIZER_NAMES))

        if not isinstance(weights, np.ndarray) or len(weights.shape) != 1:
            raise TypeError('The weights must be given as a 1d array.')
        self._weights = weights


    @property
    def weights(self):
        "Current weights."
        return self._weights


    def reg_cost_L2(self, weights, grad=False):
        """L2 regularization cost.

        Args:
          weights (array[float]): optimization parameters
          grad (bool): should we return the gradient of the regularization cost instead?

        Returns:
          float, array[float]: regularization cost or its gradient
        """
        if grad:
            # gradient with respect to each weight
            return (self._hp['lambda'] * 2) * weights
        return self._hp['lambda'] * np.sum(weights ** 2)


    def train(self, max_steps=None, batch_size=None, print_every=1):
        """Train the system.

        Args:
          max_steps (int): maximum number of steps for the algorithm
          batch_size (int): size of the data batch. Only used when 'optimizer' is 'SGD'.
          print_every (int): add a log entry every print_every steps
        """
        init_lr = self._hp["init_learning_rate"]
        decay = self._hp['decay']
        optimizer = self._hp["optimizer"]

        if batch_size is not None and batch_size > self._n_data:
            raise ValueError('Batch size cannot be larger than the total number of data samples.')

        x0 = self._weights  # initial weights
        global_step = 0

        if self._hp['regularizer'] is not None:
            err_func = lambda x, batch=None: self._cost_func(x, batch) +self.reg_cost_L2(x)
            err_grad = lambda x, batch=None: self._cost_grad(x, batch) +self.reg_cost_L2(x, grad=True)
        else:
            err_func = lambda x, batch=None: self._cost_func(x, batch)
            err_grad = lambda x, batch=None: self._cost_grad(x, batch)

        log.info('Initial cost: {}'.format(err_func(x0)))

        def signal_handler(sig, frame):
            "Called when SIGINT is received, for example when the user presses ctrl-c."
            self.stop = True

        # catch ctrl-c gracefully
        signal.signal(signal.SIGINT, signal_handler)

        if optimizer == "SGD":   # stochastic gradient descent
            x = x0.copy()
            log.info('Global step, \tCost, \tLearning rate\n')
            for step in range(global_step, global_step + max_steps):
                # generate a random batch of data samples
                if batch_size is not None:
                    perm = np.random.permutation(self._n_data)
                    batch = perm[:batch_size]
                else:
                    batch = None

                # take a step against the gradient  TODO does not ensure that the cost goes down, should it?
                grad = err_grad(x, batch)
                decayed_lr = init_lr / (1 +decay*step)
                x -= decayed_lr * grad
                cost = err_func(x)

                if step % print_every == 0:
                    log.info('{:d}, \t{:.4g}, \t{:.4g}'.format(step, cost, decayed_lr))
                if self.stop:
                    break
            self._weights = x

        elif optimizer in OPTIMIZER_NAMES:
            def callback(x):
                print('callback called')
                if self.stop:
                    raise RuntimeError('User stop.')

            opt = minimize(err_func,
                           x0,
                           method=optimizer,
                           jac=err_grad,
                           callback=callback,
                           options={"maxiter": max_steps, "disp": True})
            self._weights = opt.x

            print("Optimisation successful: ", opt.success)
            try:
                print("Number of iterations performed: ", opt.nit)
            except AttributeError:
                print("Number of iterations performed: Not applicable to solver.")
            print("Final parameters: ", opt.x)
            print("Reason for termination: ", opt.message)
        else:
            raise ValueError("Unknown optimisation method '{}'.".format(optimizer))

        print("\nFinal cost: {}\n".format(err_func(self._weights)))

        # restore default handler
        signal.signal(signal.SIGINT, signal.SIG_DFL)
