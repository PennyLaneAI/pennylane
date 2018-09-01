# Copyright 2018 Xanadu Quantum Technologies Inc.
r"""
Optimization methods
====================

**Module name:** :mod:`openqml.optimize`

.. currentmodule:: openqml.optimize


The optimizer functions can be called either directly or via the :class:`Optimizer` class (based on MLtoolbox by Maria Schuld).
Additionally, :class:`Optimizer` supports the :func:`scipy.optimize.minimize` optimization methods.


Functions
---------

.. autosummary::
   optimize_SGD


Classes
-------

.. autosummary::
   Optimizer


Optimizer methods
-----------------

.. currentmodule:: openqml.optimize.Optimizer

.. autosummary::
   print_hp
   set_hp
   weights
   train


Optimizer private methods
-------------------------

.. autosummary::
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
    "Exception for stopping the optimization prematurely."
    pass


OPTIMIZER_NAMES = ["SGD", "Nelder-Mead", "Powell", "CG", "BFGS", "Newton-CG",
                   "L-BFGS-B", "TNC", "COBYLA", "SLSQP", "dogleg", "trust-ncg",
                   "trust-exact", "trust-krylov"]



def optimize_SGD(err_func, err_grad, x0, *, data, max_steps=100, ilr=0.1, decay=0.03, batch_size=None, callback=None):
    """Stochastic Gradient Descent optimization.

    Args:
      err_func (callable): Error function. Arguments: optimization parameters (array[float]), and an array of data samples.
      err_grad (callable): Gradient of the error function with respect to the parameters. Takes the same arguments as ``err_func``.
      x0 (array[float]): initial values for the optimization parameters

    Keyword Args:
      data  (array): data samples to be used in the optimization
      max_steps (int): maximum number of iterations for the algorithm
      ilr   (float): initial learning rate, usually around 0.1
      decay (float): decay rate for the learning rate
      batch_size (int, None): How many randomly chosen data samples to include in computing the cost function each iteration. None means all.
      callback (callable, None): if given, will be called after every iteration with the current parameter values as the argument

    Returns:
      scipy.optimize.OptimizeResult: results of the optimization

    .. todo:: This is a simple SGD variant that uses a monotonously decreasing step size with no momentum.
    """
    # data batching
    n_data = data.shape[0]
    if batch_size is None:
        batch_size = n_data  # use all the data
    elif batch_size > n_data:
        raise ValueError('Batch size cannot be larger than the total number of data samples.')

    global_step = 0
    x = x0  # current weights
    success = True
    msg = 'Requested number of iterations finished.'

    for step in range(global_step, global_step + max_steps):
        # generate a random batch of data samples
        perm = np.random.permutation(n_data)
        batch = perm[:batch_size]
        batch = data[batch]

        # take a step against the gradient  TODO does not ensure that the cost goes down, should it?
        grad = err_grad(x, batch)
        decayed_lr = ilr / (1 +decay*step)
        x -= decayed_lr * grad

        # call the callback
        if callback is not None:
            cost = err_func(x, data)  # use all the data
            if callback(x, cost):
                msg = 'Optimization terminated successfully.'
                break

    return OptimizeResult({'success': success,
                           'x': x,
                           'nit': step-global_step,
                           'message': msg})



class Optimizer:
    """Quantum circuit optimizer.

    Optimization hyperparameters are given as keyword arguments.

    Args:
      cost_func (callable): Cost function. Typically involves the evaluation of one or more :class:`~openqml.circuit.QNode` instances
        representing variational quantum circuits. Arguments: weights (array[float]) and optionally an array of data items.
      cost_grad (callable): Gradient of the cost function with respect to the weights. Takes the same arguments as ``cost_func``.
        Typically obtained using autograd as :code:`cost_grad = autograd.grad(cost_func, 0)`.
      weights (array[float]): initial values for the weights/optimization parameters

    Keyword Args:
      optimizer (str): 'SGD' or any optimizer not requiring a Hessian, compatible with :func:`scipy.optimize.minimize`: 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'SLSQP'
      regularizer (None, str, callable):  None, 'L2', or a custom function mapping array[float] to float.
      reg_lambda (float): regularization strength
      print_every (int): add a status entry into the log every print_every iterations
    """
    def __str__(self):
        """String representation."""
        return self.__class__.__name__ +': ' +self._hp['optimizer']

    def __init__(self, cost_func, cost_grad, weights, **kwargs):
        self._cost_func = cost_func  #: callable: scalar function to be minimized
        self._cost_grad = cost_grad  #: callable: gradient of _cost_func
        self.stop = False            #: bool: flag, stop optimization

        self._default_hp = {'optimizer': 'SGD',
                            'reg_lambda': 0.0,
                            'regularizer': None,
                            'print_every': 5,
        }  #: dict[str->*]: default hyperparameters
        self._hp = self._default_hp.copy()  #: dict[str->*]: hyperparameters
        self._hp.update(kwargs) # update with user-given hyperparameters
        self.print_hp(log.info)

        # check the requested optimizer
        temp = self._hp['optimizer']
        if not callable(temp) and temp not in OPTIMIZER_NAMES:
            raise ValueError('The optimizer has to be either a callable or in the list of allowed optimizers, {}'.format(OPTIMIZER_NAMES))
        if temp in ['Nelder-Mead', 'Powell'] and cost_grad is not None:
            raise ValueError('{} does not use a gradient function.'.format(temp))

        if not isinstance(weights, np.ndarray) or len(weights.shape) != 1:
            raise TypeError('The weights must be given as a 1d array.')
        self._weights = weights  #: array[float]: optimization parameters

    @property
    def weights(self):
        "Current weights."
        return self._weights

    def print_hp(self, print_func=print):
        "Print the hyperparameters."
        print_func('Hyperparameters:')
        for key, value in sorted(self._hp.items()):
            temp = ''
            if key in self._default_hp:
                if self._default_hp[key] == value:
                    temp = ' (default)'
            print_func('{:16s}{!s:11s}{}'.format(key, value, temp))
        print_func('')

    def set_hp(self, **kwargs):
        """Set hyperparameter values.

        See the keyword args in :meth:`__init__`.
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
            return (self._hp['reg_lambda'] * 2) * weights
        return self._hp['reg_lambda'] * np.sum(weights ** 2)

    def train(self, max_steps=100, error_goal=None, data=None, **kwargs):
        """Optimize the system.

        Args:
          max_steps    (int): maximum number of steps for the algorithm
          error_goal (float, None): acceptable error function value (optimization finishes when it is reached), or None if a strict minimum is required
          data (array, None): data samples to be used in training, or None if no data is used. If given, cost_func and its gradient must accept a batch of data samples as the second argument.

        Additional keyword args are passed on to the optimizer function.

        Returns:
          scipy.optimize.OptimizeResult: dict subclass containing the results of the optimization run
        """
        self.stop = False

        # regularization error
        reg = self._hp['regularizer']
        if reg is None:
            reg = lambda x, grad=False: 0.0
        elif callable(reg):
            pass
        elif reg == 'L2':
            reg = self._reg_cost_L2
        else:
            raise ValueError('Unknown regularizer.')

        # prepare the error function (sum of cost and regularization) and its gradient
        if data is None:
            # no data was given, error function takes one parameter
            self.err_func = lambda x: self._cost_func(x) +reg(x)
            self.err_grad = lambda x: self._cost_grad(x) +reg(x, grad=True)
        else:
            # error function takes two parameters, weights and data batch
            self.err_func = lambda x, batch=data: self._cost_func(x, batch) +reg(x)
            self.err_grad = lambda x, batch=data: self._cost_grad(x, batch) +reg(x, grad=True)

        # if a gradient function was not supplied by the caller, assume it is not required
        if self._cost_grad is None:
            self.err_grad = None

        self.nit = 0  #: int: number of iterations performed
        x0 = self._weights  # initial weights
        log.info('Initial cost: {:10.6g}'.format(self.err_func(x0)))

        def signal_handler(sig, frame):
            "Called when SIGINT is received, for example when the user presses ctrl-c."
            self.stop = True

        # catch ctrl-c gracefully
        signal.signal(signal.SIGINT, signal_handler)

        def callback(x, err=None):
            """Callback function executed by the optimizer after every iteration.

            Returns:
              bool: True if optimization should be finished here
            """
            self._weights = x
            if err is None:
                err = self.err_func(x)  # TODO when Scipy is updated so that the callback also receives the objective function value as an argument, remove this
            if self.nit % self._hp['print_every'] == 0:
                log.info('{:13d} {:10.6g}'.format(self.nit, err))
            self.nit += 1
            if self.stop:
                raise StopOptimization('User stop.')
            if error_goal is not None and err < error_goal:
                return True

        log.info('Iteration       Cost')
        log.info('------------------------')

        optimizer = self._hp['optimizer']
        try:
            if optimizer == 'SGD':   # stochastic gradient descent
                if data is None:
                    raise ValueError('SGD requires a data sample.')
                opt = optimize_SGD(self.err_func, self.err_grad, x0, data=data, max_steps=max_steps, callback=callback, **kwargs)

            elif optimizer in OPTIMIZER_NAMES:
                opt = minimize(self.err_func, x0, method=optimizer, jac=self.err_grad, callback=callback,
                               options={'maxiter': max_steps, 'disp': True}, **kwargs)
            else:
                raise ValueError("Unknown optimization method '{}'.".format(optimizer))

        except StopOptimization as exc:
            # TODO the callback should maybe store more optimization information than just the last x
            opt = OptimizeResult({'success': False,
                                  'x': self._weights,
                                  'nit': self.nit,
                                  'message': str(exc)})
        else:
            # optimization ended on its own
            self._weights = opt.x

        print('\nOptimization successful:', opt.success)
        print('Reason for termination:', opt.message)
        try:
            print('Number of iterations performed:', opt.nit)
        except AttributeError:
            print('Number of iterations performed: Not applicable to solver.')
        print('Final parameters:', opt.x)

        opt.fun = self.err_func(self._weights)  # TODO some optimizers may return this on their own
        print('Final error: {:.6g}\n'.format(opt.fun))

        # restore default handler
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        return opt
