# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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

import numpy as np
from pennylane.measurements import Shots


class SPSAOptimizer:
    r"""The Simultaneous Perturbation Stochastic Approximation method (SPSA)
    is a stochastic approximation algorithm for optimizing cost functions whose evaluation may involve noise.

    While other gradient-based optimization methods usually attempt to compute
    the gradient analytically, SPSA involves approximating gradients at the cost of
    evaluating the cost function twice in each iteration step. This cost may result in
    a significant decrease in the overall cost of function evaluations for the entire optimization.
    It is based on an approximation of the unknown gradient :math:`\hat{g}(\hat{\theta}_{k})`
    through a simultaneous perturbation of the input parameters:

    .. math::
        \hat{g}_k(\hat{\theta}_k) = \frac{y(\hat{\theta}_k+c_k\Delta_k)-
        y(\hat{\theta}_k-c_k\Delta_k)}{2c_k} \begin{bmatrix}
           \Delta_{k1}^{-1} \\
           \Delta_{k2}^{-1} \\
           \vdots \\
           \Delta_{kp}^{-1}
         \end{bmatrix}\text{,}

    where

    * :math:`k` is the current iteration step,
    * :math:`\hat{\theta}_k` are the input parameters at iteration step :math:`k`,
    * :math:`y` is the objective function,
    * :math:`c_k=\frac{c}{k^\gamma}` is the gain sequence corresponding to evaluation step size
      and it can be controlled with

      * scaling parameter :math:`c` and
      * scaling exponent :math:`\gamma`

    * :math:`\Delta_{ki}^{-1} \left(1 \leq i \leq p \right)` are the inverted elements of
      random pertubation vector :math:`\Delta_k`.

    :math:`\hat{\theta}_k` is updated to a new set of parameters with

    .. math::
        \hat{\theta}_{k+1} = \hat{\theta}_{k} - a_k\hat{g}_k(\hat{\theta}_k)\text{,}

    where the gain sequences :math:`a_k=\frac{a}{(A+k)^\alpha}` controls parameter update step size.

    The gain sequence :math:`a_k` can be controlled with

    * scaling parameter :math:`a`,
    * scaling exponent :math:`\alpha` and
    * stability constant :math:`A`

    For more details, see `Spall (1998a)
    <https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF>`_.

    .. note::

        * One SPSA iteration step of a cost function that involves computing the expectation value of
          a Hamiltonian with ``M`` terms requires :math:`2*M` quantum device executions.
        * The forward-pass value of the cost function is not computed when stepping the optimizer.
          Therefore, in case of using ``step_and_cost`` method instead of ``step``, the number
          of executions will include the cost function evaluations.


    **Examples:**

    For VQE/VQE-like problems, the objective function can be the following:

    >>> coeffs = [0.2, -0.543, 0.4514]
    >>> obs = [qml.X(0) @ qml.Z(1), qml.Z(0) @ qml.Hadamard(2),
    ...             qml.X(3) @ qml.Z(1)]
    >>> H = qml.Hamiltonian(coeffs, obs)
    >>> num_qubits = 4
    >>> dev = qml.device("default.qubit", wires=num_qubits)
    >>> @qml.qnode(dev)
    ... def cost(params, num_qubits=1):
    ...     qml.BasisState(np.array([1, 1, 0, 0]), wires=range(num_qubits))
    ...     for i in range(num_qubits):
    ...         qml.Rot(*params[i], wires=0)
    ...         qml.CNOT(wires=[2, 3])
    ...         qml.CNOT(wires=[2, 0])
    ...         qml.CNOT(wires=[3, 1])
    ...     return qml.expval(H)
    ...
    >>> params = np.random.normal(0, np.pi, (num_qubits, 3), requires_grad=True)

    Once constructed, the cost function can be passed directly to the
    ``step`` or ``step_and_cost`` function of the optimizer:

    >>> max_iterations = 100
    >>> opt = qml.SPSAOptimizer(maxiter=max_iterations)
    >>> for _ in range(max_iterations):
    ...     params, energy = opt.step_and_cost(cost, params, num_qubits=num_qubits)
    >>> print(energy)
    -0.4294539602541956

    The algorithm provided by SPSA does not rely on built-in automatic differentiation capabilities of the interface being used
    and therefore the optimizer can be used in more complex hybrid classical-quantum workflow with any of the interfaces:

    >>> n_qubits = 1
    >>> max_iterations = 20
    >>> dev = qml.device("default.qubit", wires=n_qubits)
    >>> @qml.qnode(dev, interface="tf")
    ... def layer_fn_spsa(inputs, weights):
    ...     qml.AngleEmbedding(inputs, wires=range(n_qubits))
    ...     qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    ...     return qml.expval(qml.Z(0))
    ...
    >>> opt = qml.SPSAOptimizer(maxiter=max_iterations)
    ... def fn(params, tensor_in, tensor_out):
    ...     with tf.init_scope():
    ...             for _ in range(max_iterations):
    ...                     # Some classical steps before the quantum computation
    ...                     params_a, layer_res = opt.step_and_cost(layer_fn_spsa,
    ...                                     np.tensor(tensor_in, requires_grad=False),
    ...                                     np.tensor(params))
    ...                     params = params_a[1]
    ...                     tensor_out = layer_res
    ...                     # Some classical steps after the quantum computation
    ...     return layer_res
    ...
    >>> tensor_in = tf.Variable([0.27507603])
    >>> tensor_out = tf.Variable([0])
    >>> params = tf.Variable([[3.97507603],
    ...     [3.12950603],
    ...     [1.00854038],
    ...     [1.25907603]])
    >>> loss = fn(params, tensor_in, tensor_out)
    >>> print(loss)
    tf.Tensor(-0.9995854230771829, shape=(), dtype=float64)



    Keyword Args:
        maxiter (int): the maximum number of iterations expected to be performed.
            Used to determine :math:`A`, if :math:`A` is not supplied, otherwise ignored.
        alpha (float): A hyperparameter to calculate :math:`a_k=\frac{a}{(A+k+1)^\alpha}`
            for each iteration. Its asymptotically optimal value is 1.0.
        gamma (float): An hyperparameter to calculate :math:`c_k=\frac{c}{(k+1)^\gamma}`
            for each iteration. Its asymptotically optimal value is 1/6.
        c (float): A hyperparameter related to the expected noise. It should be
            approximately the standard deviation of the expected noise of the cost function.
        A (float): The stability constant; if not provided, set to be 10% of the maximum number
            of expected iterations.
        a (float): A hyperparameter expected to be small in noisy situations,
            its value could be picked using `A`, :math:`\alpha` and :math:`\hat{g_0} (\hat{\theta_0})`.
            For more details, see `Spall (1998b)
            <https://www.jhuapl.edu/spsa/PDF-SPSA/Spall_Implementation_of_the_Simultaneous.PDF>`_.
    """

    # pylint: disable-msg=too-many-arguments
    def __init__(self, maxiter=None, alpha=0.602, gamma=0.101, c=0.2, A=None, a=None):
        self.a = a
        self.A = A
        if not maxiter and not A:
            raise TypeError("One of the parameters maxiter or A must be provided.")
        if not A:
            self.A = maxiter * 0.1
        if not a:
            self.a = 0.05 * (self.A + 1) ** alpha
        self.c = c
        self.alpha = alpha
        self.gamma = gamma
        self.k = 1
        self.ak = self.a / (self.A + 1) ** self.alpha

    def step_and_cost(self, objective_fn, *args, **kwargs):
        r"""Update the parameter array :math:`\hat{\theta}_k` with one step of the
        optimizer and return the step and the corresponding objective function. The number
        of steps stored by the ``k`` attribute of the optimizer is counted internally when calling ``step_and_cost`` and ``cost``.

        Args:
            objective_fn (function): the objective function for optimization
            *args : variable length argument array for objective function
            **kwargs : variable length of keyword arguments for the objective function

        Returns:
            tuple[list [array], float]: the new variable values :math:`\hat{\theta}_{k+1}` and the
            objective function output prior to the step.
        """
        g = self.compute_grad(objective_fn, args, kwargs)
        new_args = self.apply_grad(g, args)

        self.k += 1

        forward = objective_fn(*args, **kwargs)

        # unwrap from list if one argument, cleaner return
        if len(new_args) == 1:
            return new_args[0], forward
        return new_args, forward

    def step(self, objective_fn, *args, **kwargs):
        r"""Update trainable arguments with one step of the optimizer. The number
        of steps is being counted through calls to ``step_and_cost`` and ``cost``.

        Args:
            objective_fn (function): the objective function for optimization
            *args : variable length argument array for objective function
            **kwargs : variable length of keyword arguments for the objective function

        Returns:
            list [array]: the new variable values :math:`\hat{\theta}_{k+1}`.
        """
        g = self.compute_grad(objective_fn, args, kwargs)
        new_args = self.apply_grad(g, args)

        self.k += 1

        # unwrap from list if one argument, cleaner return
        if len(new_args) == 1:
            return new_args[0]

        return new_args

    def compute_grad(self, objective_fn, args, kwargs):
        r"""Approximate the gradient of the objective function at the
        given point.

        Args:
            objective_fn (function): The objective function for optimization
            args (tuple): tuple of NumPy array containing the current parameters
                for objective function
            kwargs (dict): keyword arguments for the objective function

        Returns:
            tuple (array): NumPy array containing the gradient
                :math:`\hat{g}_k(\hat{\theta}_k)`
        """
        ck = self.c / self.k**self.gamma

        delta = []
        thetaplus = list(args)
        thetaminus = list(args)

        for index, arg in enumerate(args):
            if getattr(arg, "requires_grad", False):
                # Use the symmetric Bernoulli distribution to generate
                # the coordinates of delta. Note that other distributions
                # may also be used (they need to satisfy certain conditions).
                # Refer to the paper linked in the class docstring for more info.
                di = np.random.choice([-1, 1], size=arg.shape)
                multiplier = ck * di
                thetaplus[index] = arg + multiplier
                thetaminus[index] = arg - multiplier
                delta.append(di)
        yplus = objective_fn(*thetaplus, **kwargs)
        yminus = objective_fn(*thetaminus, **kwargs)
        try:
            # pylint: disable=protected-access
            dev_shots = objective_fn.device.shots
            if isinstance(dev_shots, Shots):
                shots = dev_shots if dev_shots.has_partitioned_shots else Shots(None)
            elif objective_fn.device.shot_vector is not None:
                shots = Shots(objective_fn.device._raw_shot_sequence)  # pragma: no cover
            else:
                shots = Shots(None)
            if np.prod(objective_fn.func(*args).shape(objective_fn.device, shots)) > 1:
                raise ValueError(
                    "The objective function must be a scalar function for the gradient "
                    "to be computed."
                )
        except AttributeError:
            if yplus.size > 1:
                raise ValueError(  # pylint: disable=raise-missing-from
                    "The objective function must be a scalar function for the gradient "
                    "to be computed."
                )
        grad = [(yplus - yminus) / (2 * ck * di) for di in delta]

        return tuple(grad)

    def apply_grad(self, grad, args):
        r"""Update the variables to take a single optimization step.

        Args:
            grad (tuple [array]): the gradient approximation of the objective
                function at point :math:`\hat{\theta}_{k}`
            args (tuple): the current value of the variables :math:`\hat{\theta}_{k}`

        Returns:
            list [array]: the new values :math:`\hat{\theta}_{k+1}`"""
        self.ak = self.a / (self.A + self.k) ** self.alpha
        args_new = list(args)
        trained_index = 0
        for index, arg in enumerate(args):
            if getattr(arg, "requires_grad", False):
                args_new[index] = arg - self.ak * grad[trained_index]
                trained_index += 1

        return args_new
