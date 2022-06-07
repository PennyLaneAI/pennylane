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

from pennylane import numpy as np


class SPSAOptimizer:
    r"""The Simultaneous Perturbation Stochastic Approximation method (SPSA)
    is a stochastic approximation algorithm for optimizing cost functions whose evaluation may involve noise.

    While other gradient-based optimization methods usually attempt to compute
    the gradient analytically, SPSA involves approximating gradients at the cost of
    evaluating the cost function twice in each iteration step. This cost may result in
    a significant decrease in the overall cost of function evaluations for the entire optimization.
    It is based on an approximation to the unknown gradient :math:`\hat{g}(\hat{\theta}_{k})`
    through a simultaneous perturbation of the input parameters:

    .. math::
        \hat{g}_k(\hat{\theta}_k) = \frac{y(\hat{\theta}_k+c_k\Delta_k)-
        y(\hat{\theta}_k-c_k\Delta_k)}{2c_k} \begin{bmatrix}
           \Delta_{k1}^{-1} \\
           \Delta_{k2}^{-1} \\
           \vdots \\
           \Delta_{kp}^{-1}
         \end{bmatrix}

    To update :math:`\hat{\theta}_k` to a new set of parameters:

    .. math::
        \hat{\theta}_{k+1} = \hat{\theta}_{k} - a_k\hat{g}_k(\hat{\theta}_k)

    where the gain sequences are :math:`a_k=\frac{a}{(A+k+1)^\alpha}` and
    :math:`c_k=\frac{c}{(k+1)^\gamma}`


    For more details, see:

        J. Spall
        "An Overview of the Simultaneous Perturbation Method for Efficient Optimization."
        `Johns Hopkins api technical digest, 19(4), 482-492, 1998
        <https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF>`_

    .. note::

        * The number of quantum device executions is :math:`2*iter*num terms hamiltonian`.
        * The forward-pass value of the cost function is not computed when stepping the optimizer.
        Therefore, in case of using ``step_and_cost`` method instead of ``step``, the number
          of executions will include the cost function evaluations.

    .. note::

        In cases of hybrid classical-quantum workflows:

        * In implementation of a QNode as a layer of a Keras sequential model,
          possible optimizers for the model are from the classical platform i.e.
          `tf.keras.optimizers.SGD
          <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD>`_.
        * In a hybrid classical-quantum-classical workflow where we use SPSAOptimizer
          for the quantum part, we have to extract the values of the classical tensor
          in order to use it in the quantum circuit as inputs and assign the output
          of the quantum circuit to a subsecuent classical tensor.


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
    ...     params, energy = opt.step_and_cost(cost, params)

    Example of hybrid classical-quantum workflow:

    >>> dev = qml.device("default.qubit", wires=n_qubits)
    >>> @qml.qnode(dev, interface=None)
    >>> def layer_fn_spsa(inputs, weights):
    ...     qml.AngleEmbedding(inputs, wires=range(n_qubits))
    ...     qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    ...     return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    >>> opt = qml.SPSAOptimizer(maxiter=max_iterations)
    >>> tensor_in = tf.Variable([0.27507603, 0.3453423])
    >>> params = tf.Variable([[3.97507603, 2.00854038],
    ...                       [3.12950603, 3.00854038],
    ...                       [1.17907603, 1.10854038],
    ...                       [0.97507603, 1.00854038],
    ...                       [1.25907603, 0.40854088]])
    >>> tensor_out = tf.Variable([0,0])

    >>> init = tf.compat.v1.global_variables_initializer()
    >>> with tf.compat.v1.Session() as sess:
    ...     sess.run(init)
    ...     for _ in range(max_iterations):
    ...         # Take step
    ...         params_a, layer_res = opt.step_and_cost(layer_fn_spsa,
    ...                                np.tensor(tensor_in.eval(sess), requires_grad=False),
    ...                                np.tensor(params.eval(sess)))
    ...         params.assign(params_a[1], sess)
    ...         tensor_out.assign(layer_res, sess)




    Keyword Args:
        maxiter (int): the maximum number of iterations expected to be performed
        alpha (float): A hyperparameter to calculate :math:`a_k=\frac{a}{(A+k+1)^\alpha}`
            for each iteration. Its asymptotically optimal value is 1.0.
        gamma (float): An hyperparameter to calculate :math:`c_k=\frac{c}{(k+1)^\gamma}`
            for each iteration. Its asymptotically optimal value is 1/6.
        c (float): An hyperparameter related to the expected noise. It should be
            approximately the standard deviation of the expected noise on the cost function
        A (float): The stability constant expected to be 10% of maximum number
            of expected iterations
        a (float): An hyperparameter expected to be small in noisy situations,
            whose value could be :math:`\frac{mag(\Delta\theta)}{mag(g(\theta))}(A+1)^\alpha`
    """
    # pylint: disable-msg=too-many-arguments
    def __init__(self, maxiter=200, alpha=0.602, gamma=0.101, c=0.2, A=None, a=None):
        if not A:
            self.A = maxiter * 0.1
        if not a:
            self.a = 0.05 * (self.A + 1) ** alpha
        self.c = c
        self.alpha = alpha
        self.gamma = gamma
        self.k = 0
        self.ak = self.a / (self.A + 1 + 1.0) ** self.alpha

    def step_and_cost(self, objective_fn, *args, **kwargs):
        """Update the parameter array :math:`x` with one step of the optimizer and return
        the step and the corresponding objective function.

        Args:
            objective_fn (function): The objective function for optimization
            *args : variable length argument array for objective function
            **kwargs : variable length of keyword arguments for the objective function

        Returns:
            tuple[list [array], float]: the new variable values :math:`x^{(t+1)}` and the
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
            list [array]: the new variable values :math:`x^{(t+1)}`.
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
            args (tuple): tuple of NumPy array containing the current parameters
                for objective function
            kwargs (dict): keyword arguments for the objective function

        Returns:
            tuple (array): Numpy array containing the gradient
                :math:`\hat{g}_k(\hat{\theta}_k)` and ``None``
        """
        self.increment_k()
        ck = self.c / (self.k + 1.0) ** self.gamma
        delta = list(args)
        thetaplus = list(args)
        thetaminus = list(args)
        grad = []
        for index, arg in enumerate(args):
            if getattr(arg, "requires_grad", False):
                delta[index] = np.random.choice([-1, 1], size=arg.shape)
                thetaplus[index] = arg + ck * delta[index]
                thetaminus[index] = arg - ck * delta[index]
        yplus = objective_fn(*thetaplus, **kwargs)
        yminus = objective_fn(*thetaminus, **kwargs)
        for index, arg in enumerate(args):
            if getattr(arg, "requires_grad", False):
                grad.append((yplus - yminus) / (2 * ck * delta[index]))

        return tuple(grad), None

    def apply_grad(self, grad, args):
        r"""Update the variables to take a single optimization step.

        Args:
            grad (tuple [array]): the gradient approximation of the objective
                function at point :math:`x^{(t)}`
            args (tuple): the current value of the variables :math:`x^{(t)}`

        Returns:
            list [array]: the new values :math:`x^{(t+1)}`"""
        self.ak = self.a / (self.A + self.k + 1.0) ** self.alpha
        args_new = list(args)
        trained_index = 0
        for index, arg in enumerate(args):
            if getattr(arg, "requires_grad", False):
                args_new[index] = arg - self.ak * grad[trained_index]
                trained_index += 1

        return args_new
