# Copyright 2018 Xanadu Quantum Technologies Inc.
r"""
Core classes
============

**Module name:** :mod:`openqml.core`

.. currentmodule:: openqml.core


The :class:`Optimizer` class is based on MLtoolbox by Maria Schuld.



Classes
-------

.. autosummary::
   Optimizer


Optimizer methods
-----------------

.. currentmodule:: openqml.core.Optimizer

.. autosummary::
   set_hp
   weights
   train


Optimizer private methods
-------------------------

.. autosummary::
   _optimize_SGD
   _reg_cost_L2

----
"""

import signal
import logging as log

import numpy as np
from scipy.optimize import minimize, OptimizeResult


# optimization parameters
#Par = namedtuple('Par', 'name, init, regul')
#Par.__new__.__defaults__ = ('', 0, False)


class StopOptimization(Exception):
    "Exception for stopping the optimization."
    pass


OPTIMIZER_NAMES = ["SGD", "Nelder-Mead", "Powell", "CG", "BFGS", "Newton-CG",
                   "L-BFGS-B", "TNC", "COBYLA", "SLSQP", "dogleg", "trust-ncg",
                   "trust-exact", "trust-krylov"]


class Optimizer:
    """Quantum circuit optimizer.

    Optimization hyperparameters are given as keyword arguments.

    Args:
      cost_func (callable): Cost/error function. Typically involves the evaluation of one or more :class:`~openqml.circuit.QNode` instances
        representing variational quantum circuits. Takes two arguments: weights (array[float]) and optionally a batch of data item indices (Sequence[int]).
      cost_grad (callable): Gradient of the cost/error function with respect to the weights. Takes the same argumets as ``cost_func``.
        Typically obtained using autograd as :code:`cost_grad = autograd.grad(cost_func, 0)`.
      weights (array[float]): initial values for the weights/optimization parameters
      n_data (int): total number of data samples to be used in training

    Keyword Args:
      optimizer (str): 'SGD' or any optimizer not requiring a Hessian, compatible with :func:`scipy.optimize.minimize`: 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'SLSQP'
      lambda (float): regularization strength
      regularizer (None, callable):  None, 'L2', or a custom function mapping Sequence[float] to float.
      init_learning_rate (float): SGD only: initial learning rate, usually around 0.1
      decay  (float): SGD only: decay rate for the learning rate
      batch_size (None, int): SGD only: How many randomly chosen data samples to include in computing the cost function each iteration. None means all of them.
      print_every (int): add a status entry into the log every print_every iterations
    """
    def __str__(self):
        """String representation."""
        return self.__class__.__name__

    def __init__(self, cost_func, cost_grad, weights, *, n_data=0, **kwargs):

        self._cost_func = cost_func  #: callable: scalar function to be minimized
        self._cost_grad = cost_grad  #: callable: gradient of _cost_func
        self._n_data = n_data        #: int: total number of data samples to be used in training
        self.stop = False            #: bool: flag, stop optimization

        # default hyperparameters
        default_hp = {'optimizer': 'SGD',
                      'init_learning_rate': 0.1,
                      'decay': 0.03,
                      'lambda': 0.0,
                      'regularizer': None,
                      'batch_size': None,
                      'print_every': 5,
        }
        self._hp = default_hp    #: dict[str->*]: hyperparameters
        self._hp.update(kwargs)  # update with user-given hyperparameters
        print("HYPERPARAMETERS:\n")
        for key in sorted(self._hp):
            temp = '' if key in kwargs else ' (default)'
            print('{:20s}{!s:10s}{}'.format(key, self._hp[key], temp))
        print()

        temp = self._hp['optimizer']
        if not callable(temp) and temp not in OPTIMIZER_NAMES:
            raise ValueError("The optimizer has to be either a callable or in the list of allowed optimizers, {}".format(OPTIMIZER_NAMES))
        if temp in ['Nelder-Mead', 'Powell'] and cost_grad is not None:
            raise ValueError("{} does not use a gradient function.".format(temp))

        if not isinstance(weights, np.ndarray) or len(weights.shape) != 1:
            raise TypeError('The weights must be given as a 1d array.')
        self._weights = weights  #: array[float]: optimization parameters


    @property
    def weights(self):
        "Current weights."
        return self._weights


    def set_hp(self, **kwargs):
        """Set hyperparameter values.
        """
        self._hp.update(kwargs)


    def _reg_cost_L2(self, weights, grad=False):
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


    def _optimize_SGD(self, x0, max_steps):
        """Stochastic Gradient Descent optimization.

        Args:
          x0 (array[float]): initial values for the optimization parameters
          max_steps (int): maximum number of iterations for the algorithm
        """
        init_lr = self._hp['init_learning_rate']
        decay = self._hp['decay']
        batch_size = self._hp['batch_size']
        print_every = self._hp['print_every']

        if batch_size is not None and batch_size > self._n_data:
            raise ValueError('Batch size cannot be larger than the total number of data samples.')

        global_step = 0
        x = x0
        success = True
        msg = 'Requested number of iterations finished.'

        log.info('Global step       Cost  Learn. rate')
        log.info('-----------------------------------')
        for step in range(global_step, global_step + max_steps):
            # generate a random batch of data samples
            if batch_size is not None:
                perm = np.random.permutation(self._n_data)
                batch = perm[:batch_size]
            else:
                batch = None

            # take a step against the gradient  TODO does not ensure that the cost goes down, should it?
            grad = self.err_grad(x, batch)
            decayed_lr = init_lr / (1 +decay*step)
            x -= decayed_lr * grad
            cost = self.err_func(x)

            #self._weights = x  # store the current weights
            if step % print_every == 0:
                log.info('{:11d} {:10.6g} {:12.6g}'.format(step, cost, decayed_lr))
            if self.stop:
                success = False
                msg = 'User stop.'
                break

        return OptimizeResult({'success': success,
                               'x': x,
                               'nit': step-global_step,
                               'message': msg})


    def train(self, max_steps=100):
        """Optimize the system.

        Args:
          max_steps (int): maximum number of steps for the algorithm
        Returns:
          float: final cost function value
        """
        if self._hp['regularizer'] is None:
            self.err_func = lambda x, batch=None: self._cost_func(x, batch)
            self.err_grad = lambda x, batch=None: self._cost_grad(x, batch)
        else:
            self.err_func = lambda x, batch=None: self._cost_func(x, batch) +self._reg_cost_L2(x)
            self.err_grad = lambda x, batch=None: self._cost_grad(x, batch) +self._reg_cost_L2(x, grad=True)

        if self._cost_grad is None:
            self.err_grad = None

        x0 = self._weights  # initial weights
        log.info('Initial cost: {:.6g}'.format(self.err_func(x0)))

        def signal_handler(sig, frame):
            "Called when SIGINT is received, for example when the user presses ctrl-c."
            self.stop = True

        # catch ctrl-c gracefully
        signal.signal(signal.SIGINT, signal_handler)

        optimizer = self._hp['optimizer']
        try:
            if optimizer == 'SGD':   # stochastic gradient descent
                opt = self._optimize_SGD(x0, max_steps)

            elif optimizer in OPTIMIZER_NAMES:
                print_every = self._hp['print_every']
                self.nit = 0  #: int: number of iterations performed
                def callback(x):
                    self._weights = x
                    if self.nit % print_every == 0:
                        log.info('{:9d} {:10.6g}'.format(self.nit, self.err_func(x)))
                    self.nit += 1
                    if self.stop:
                        raise StopOptimization('User stop.')

                log.info('Iteration       Cost')
                log.info('--------------------')
                opt = minimize(self.err_func, x0, method=optimizer, jac=self.err_grad, callback=callback,
                               options={'maxiter': max_steps, 'disp': True})
            else:
                raise ValueError("Unknown optimisation method '{}'.".format(optimizer))

        except StopOptimization as exc:
            # TODO the callback should maybe store more optimization information than just the last x
            print("\nOptimisation successful: False")
            print("Number of iterations performed: ", self.nit)
            print("Reason for termination: ", exc)
        else:
            self._weights = opt.x

            print("\nOptimisation successful: ", opt.success)
            try:
                print("Number of iterations performed: ", opt.nit)
            except AttributeError:
                print("Number of iterations performed: Not applicable to solver.")
            print("Final parameters: ", opt.x)
            print("Reason for termination: ", opt.message)

        cost = self.err_func(self._weights)
        print('\nFinal cost: {:.6g}\n'.format(cost))

        # restore default handler
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        return cost
