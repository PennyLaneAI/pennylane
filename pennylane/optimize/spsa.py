# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SPSA optimizer"""

from pennylane import numpy as np


class SPSAOptimizer:
    r"""The Simultaneous Perturbation Stochastic Approximation method (SPSA)
    is an iterative algortihm for optimization where the input information
    may be contaminated with noise.
    In contrast to other methods that perform multiple operations to determine
    the gradient, SPSA only measures two times the loss function to obtain it.
    It is based on an approximation to the unknown gradient :math:`\hat{g}(\hat{\theta}_{k})`
    through a simultaneous perturbation:

    .. math::
        \hat{g}_k(\hat{\theta}_k) = \frac{y(\hat{\theta}_k+c_k\Delta_k)-y(\hat{\theta}_k-c_k\Delta_k)}{2c_k\Delta_k} \begin{bmatrix}
           \Delta_{k1}^{-1} \\
           \Delta_{k2}^{-1} \\
           \vdots \\
           \Delta_{kp}^{-1}
         \end{bmatrix}

    To update :math:`\hat{\theta}_k` to a new set of parameters:

    .. math::
        \hat{\theta}_{k+1} = \hat{\theta}_{k} - a_k\hat{g}_k(\hat{\theta}_k)

    where the gain sequences are :math:`a_k=\frac{a}{(A+k+1)^\alpha}` and :math:`c_k=\frac{c}{(k+1)^\gamma}`


    For more details, see:

        J. Spall
        "An Overview of the Simultaneous Perturbation Method for Efficient Optimization."
        `Johns Hopkins api technical digest, 19(4), 482-492, 1998 <https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF>`_

    .. note::

        The number of quantum device executions is :math:`iter*2*num_terms_hamiltonian`.
        In case of using ``step_and_cost`` method instead of ``step``,
        the number of executions increment to calculate the cost function.



    **Examples:**

    For VQE/VQE-like problems, the objective function can be the following:

    >>> dev = qml.device("default.qubit", wires=1)
    >>> def circuit(params, wires):
    ...    qml.BasisState(np.array([1, 1, 0, 0]), wires=wires)
    ...    for i in wires:
    ...        qml.Rot(*params[i], wires=i)
    ...    qml.CNOT(wires=[2, 3])
    ...    qml.CNOT(wires=[2, 0])
    ...    qml.CNOT(wires=[3, 1])
    >>> def exp_val_circuit(params):
    ...    circuit(params, range(dev.num_wires))
    ...    return qml.expval(h2_ham)
    >>> params = np.random.normal(0, np.pi, (num_qubits, 3), requires_grad=True)
    >>> cost = qml.QNode(exp_val_circuit, dev)

    Once constructed, the cost function can be passed directly to the
    optimizer's ``step`` or ``step_and_cost`` function:

    >>> max_iterations = 100
    >>> opt = qml.SPSAOptimizer(maxiter=max_iterations)
    >>> for n in range(max_iterations):
    >>>     params, energy = opt.step_and_cost(cost, params)


    Keyword Args:
        maxiter=200 (int): the maximum number of iterations expected to be performed
        alpha (float): An hyperparameter to calculate :math:`a_k=\frac{a}{(A+k+1)^\alpha}` for each iteration. Its asymptotically optimal value is 1.0
        gamma=0.101 (float): An hyperparameter to calculate :math:`c_k=\frac{c}{(k+1)^\gamma}` for each iteration. Its asymptotically optimal value is 1/6
        c=0.2 (float): An hyperparameter related to the expected noise. It
        should be approximately the standard deviation of the expected noise on the cost function
        A=None (float): The stability constant expected to be 10% of maximum number of expected iterations
        a=None (float): An hyperparameter expected to be small in noisy situations, whose value could be :math:`\frac{mag(\Delta\theta)}{mag(g(\theta))}(A+1)^\alpha`
    """
    #pylint: disable-msg=too-many-arguments
    def __init__(self, maxiter=200, alpha=0.602, gamma=0.101, c=0.2,
                 A=None, a=None):
        if not A:
            self.A = maxiter * 0.1
        if not a:
            self.a = 0.05 * (self.A + 1)**alpha
        self.c = c
        self.alpha = alpha
        self.gamma = gamma
        self.k = 0
        self.ak = self.a / (self.A + 1 + 1.0)**self.alpha

    def step_and_cost(self, objective_fn, *args, **kwargs):
        """Update the parameter array :math:`x` with one step of the optimizer and return
        the step and the corresponding objective function

        Args:
            objective_fn (function): The objective function for optimization
            *args : variable length argument array for objective function
            **kwargs : variable length of keyword arguments for the objective function

        Returns:
            tuple[array, float]: the new variable values :math:`x^{(t+1)}` and the
            objective function output prior to the step.
        """
        g, forward = self.compute_grad(objective_fn, args, kwargs)
        new_args = self.apply_grad(g, args)

        if forward is None:
            forward = objective_fn(*args, **kwargs)

        # unwrap from list if one argument, cleaner return
        if len(new_args) == 1:
            return new_args[0], forward
        return new_args, forward

    def step(self, objective_fn, *args, **kwargs):
        """Update trainable arguments with one step of the optimizer.

        Args:
            objective_fn (function): The objective function for optimization
            *args : variable length argument array for objective function
            **kwargs : variable length of keyword arguments for the objective function

        Returns:
            array: the new variable values :math:`x^{(t+1)}`.
            """
        g, _ = self.compute_grad(objective_fn, args, kwargs)
        new_args = self.apply_grad(g, args)

        # unwrap from list if one argument, cleaner return
        if len(new_args) == 1:
            return new_args[0]

        return new_args

    def increment_k(self):
        """Increments k"""
        self.k += 1

    def compute_grad(self, objective_fn, args, kwargs):
        r"""Compute approximation of gradient of the objective function at the
        given point.

        Args:
            objective_fn (function): The objective function for optimization
            args (array): NumPy array containing the current parameters for objective function
            kwargs (dict): keyword arguments for the objective function

        Returns:
            tuple (array): Numpy array containing the gradient :math:`\hat{g}_k(\hat{\theta}_k)` and ``None``
            """
        # pylint: disable=arguments-differ
        if type(args) in [list, int, float] or len(list(np.extract_tensors(args))) > 1:
            raise ValueError("The parameters must be in a tensor.")
        self.increment_k()
        ck = self.c / (self.k + 1.0)**self.gamma
        shape = args[0].shape if isinstance(args, tuple) else args.shape
        delta = np.random.choice([-1, 1], size=shape)
        thetaplus = args + ck*delta
        thetaminus = args - ck*delta
        yplus = objective_fn(*thetaplus, **kwargs)
        yminus = objective_fn(*thetaminus, **kwargs)
        grad = (yplus-yminus) / (2*ck*delta)
        num_trainable_args = sum(getattr(arg, "requires_grad", False) for arg in args)
        grad = (grad,) if num_trainable_args == 1 else grad

        return grad, None


    def apply_grad(self, grad, args):
        r"""Update the variables to take a single optimization step.

        Args:
            grad (tuple [array]): the gradient approximation of the objective
                function at point :math:`x^{(t)}`
            args (array): the current value of the variables :math:`x^{(t)}`

        Returns:
            list [array]: the new values :math:`x^{(t+1)}`"""
        self.ak = self.a / (self.A + self.k + 1.0)**self.alpha
        args_new = list(args)
        trained_index = 0
        for index, arg in enumerate(args):
            if getattr(arg, "requires_grad", False):
                args_new[index] = arg - self.ak * grad[trained_index]
                trained_index += 1

        return args_new
