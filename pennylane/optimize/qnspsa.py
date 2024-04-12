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
"""Quantum natural SPSA optimizer"""
import warnings
from scipy.linalg import sqrtm
from pennylane import numpy as pnp
import pennylane as qml


class QNSPSAOptimizer:
    r"""Quantum natural SPSA (QNSPSA) optimizer. QNSPSA is a second-order SPSA algorithm, which
    updates the ansatz parameters with the following equation:

    .. math::

        \mathbf{x}^{(t + 1)} = \mathbf{x}^{(t)} -
        \eta \widehat{\mathbf{g}}^{-1}(\mathbf{x}^{(t)})\widehat{\nabla f}(\mathbf{x}^{(t)}),

    where :math:`f(\mathbf{x})` is the objective function with input parameters :math:`\mathbf{x}`,
    while :math:`\nabla f` is the gradient, :math:`\mathbf{g}` is the second-order Fubini-Study metric
    tensor. With QNSPSA algorithm, both the gradient and the metric tensor are estimated
    stochastically, with :math:`\widehat{\nabla f}` and :math:`\widehat{\mathbf{g}}`. This stochastic
    approach requires only a fixed number of circuit executions per optimization step,
    independent of the problem size. This preferred scaling makes
    it a promising candidate for the optimization tasks for high-dimensional ansatzes. On the
    other hand, the introduction of the Fubini-Study metric into the optimization helps to find
    better minima and allows for faster convergence.

    The gradient is estimated similarly as the `SPSA optimizer
    <https://pennylane.readthedocs.io/en/stable/code/api/pennylane.SPSAOptimizer.html>`_, with a
    pair of perturbations:

    .. math::

        \widehat{\nabla f}(\mathbf{x}) = \widehat{\nabla f}(\mathbf{x}, \mathbf{h})
        \approx \frac{1}{2\epsilon}\big(f(\mathbf{x} + \epsilon \mathbf{h}) - f(\mathbf{x} - \epsilon \mathbf{h})\big),

    where :math:`\epsilon` is the finite-difference step size specified by the user, and
    :math:`\mathbf{h}` is a randomly sampled direction vector to perform the perturbation.

    The Fubini-Study metric tensor is estimated with another two pairs of perturbations along
    randomly sampled directions :math:`\mathbf{h_1}` and :math:`\mathbf{h_2}`:

    .. math::

        \widehat{\mathbf{g}}(\mathbf{x}) = \widehat{\mathbf{g}}(\mathbf{x}, \mathbf{h}_1, \mathbf{h}_2)
        \approx \frac{\delta F}{8 \epsilon^2}\Big(\mathbf{h}_1 \mathbf{h}_2^\intercal + \mathbf{h}_2 \mathbf{h}_1^\intercal\Big),

    where :math:`F(\mathbf{x}', \mathbf{x}) = \bigr\rvert\langle \phi(\mathbf{x}') | \phi(\mathbf{x}) \rangle \bigr\rvert ^ 2`
    measures the state overlap between :math:`\phi(\mathbf{x}')` and :math:`\phi(\mathbf{x})`,
    where :math:`\phi` is the parameterized ansatz. The finite difference :math:`\delta F` is
    computed from the two perturbations:

    .. math::

        \delta F = F(\mathbf{x, \mathbf{x} + \epsilon \mathbf{h}_1} + \epsilon \mathbf{h}_2)
        - F (\mathbf{x, \mathbf{x} + \epsilon \mathbf{h}_1}) - F(\mathbf{x, \mathbf{x}
        - \epsilon \mathbf{h}_1} + \epsilon \mathbf{h}_2)
        + F(\mathbf{x, \mathbf{x} - \epsilon \mathbf{h}_1}).

    For more details, see:

        Julien Gacon, Christa Zoufal, Giuseppe Carleo, and Stefan Woerner.
        "Simultaneous Perturbation Stochastic Approximation of the Quantum Fisher Information."
        `Quantum, 5, 567 <https://quantum-journal.org/papers/q-2021-10-20-567/>`_, 2021.

    You can also find a walkthrough of the implementation in this `tutorial <https://pennylane.ai/qml/demos/qnspsa.html>`_.

    **Examples:**

    For VQE/VQE-like problems, the objective function can be defined within a qnode:

    >>> num_qubits = 2
    >>> dev = qml.device("default.qubit", wires=num_qubits)
    >>> @qml.qnode(dev)
    ... def cost(params):
    ...     qml.RX(params[0], wires=0)
    ...     qml.CRY(params[1], wires=[0, 1])
    ...     return qml.expval(qml.Z(0) @ qml.Z(1))

    Once constructed, the qnode can be passed directly to the ``step`` or ``step_and_cost``
    function of the optimizer.

    >>> from pennylane import numpy as pnp
    >>> params = pnp.random.rand(2)
    >>> opt = QNSPSAOptimizer(stepsize=5e-2)
    >>> for i in range(51):
    >>>     params, loss = opt.step_and_cost(cost, params)
    >>>     if i % 10 == 0:
    ...         print(f"Step {i}: cost = {loss:.4f}")
    Step 0: cost = 0.9987
    Step 10: cost = 0.9841
    Step 20: cost = 0.8921
    Step 30: cost = 0.0910
    Step 40: cost = -0.9369
    Step 50: cost = -0.9984

    Keyword Args:
        stepsize (float): the user-defined hyperparameter :math:`\eta` for learning rate (default: 1e-3)
        regularization (float): regularitzation term :math:`\beta` to the Fubini-Study metric tensor
            for numerical stability (default: 1e-3)
        finite_diff_step (float): step size :math:`\epsilon` to compute the finite difference
            gradient and the Fubini-Study metric tensor (default: 1e-2)
        resamplings (int): the number of samples to average for each parameter update (default: 1)
        blocking (boolean): when set to be True, the optimizer only accepts updates that lead to a
            loss value no larger than the loss value before update, plus a tolerance. The tolerance
            is set with the hyperparameter ``history_length``. The ``blocking`` option is
            observed to help the optimizer to converge significantly faster (default: True)
        history_length (int): when ``blocking`` is True, the tolerance is set to be the average of
            the cost values in the last ``history_length`` steps (default: 5)
        seed (int): seed for the random sampling (default: None)
    """

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        stepsize=1e-3,
        regularization=1e-3,
        finite_diff_step=1e-2,
        resamplings=1,
        blocking=True,
        history_length=5,
        seed=None,
    ):
        self.stepsize = stepsize
        self.reg = regularization
        self.finite_diff_step = finite_diff_step
        self.metric_tensor = None
        self.k = 1
        self.resamplings = resamplings
        self.blocking = blocking
        self.last_n_steps = pnp.zeros(history_length)
        self.rng = pnp.random.default_rng(seed)

    def step(self, cost, *args, **kwargs):
        """Update trainable arguments with one step of the optimizer.

        .. note::
            When blocking is set to be True, ``step`` calls ``step_and_cost`` on the backend, as loss
            measurements are required by the algorithm in this scenario.

        Args:
            cost (QNode): the QNode wrapper for the objective function for optimization
            args : variable length argument list for qnode
            kwargs : variable length of keyword arguments for the qnode

        Returns:
            pnp.array: the new variable values after step-wise update :math:`x^{(t+1)}`
        """
        if self.blocking:
            warnings.warn(
                "step_and_cost() instead of step() is called when "
                "blocking is turned on, as the step-wise loss value "
                "is required by the algorithm.",
                stacklevel=2,
            )
            return self.step_and_cost(cost, *args, **kwargs)[0]

        return self._step_core(cost, args, kwargs)

    def step_and_cost(self, cost, *args, **kwargs):
        r"""Update trainable parameters with one step of the optimizer and return
        the corresponding objective function value after the step.

        Args:
            cost (QNode): the QNode wrapper for the objective function for optimization
            args : variable length argument list for qnode
            kwargs : variable length of keyword arguments for the qnode

        Returns:
            (np.array, float): the new variable values :math:`x^{(t+1)}` and the objective
            function output prior to the step
        """
        params_next = self._step_core(cost, args, kwargs)

        if not self.blocking:
            loss_curr = cost(*args, **kwargs)
            return params_next, loss_curr

        params_next, loss_curr = self._apply_blocking(cost, args, kwargs, params_next)
        return params_next, loss_curr

    def _step_core(self, cost, args, kwargs):
        """Core step function that returns the updated parameter before blocking condition
        is applied.

        Args:
            cost (QNode): the QNode wrapper for the objective function for optimization
            args : variable length argument list for qnode
            kwargs : variable length of keyword arguments for the qnode

        Returns:
            pnp.array: the new variable values :math:`x^{(t+1)}` before the blocking condition
            is applied.
        """
        all_grad_tapes = []
        all_grad_dirs = []
        all_metric_tapes = []
        all_tensor_dirs = []
        for _ in range(self.resamplings):
            # grad_tapes contains 2 tapes for the gradient estimation
            grad_tapes, grad_dirs = self._get_spsa_grad_tapes(cost, args, kwargs)
            # metric_tapes contains 4 tapes for tensor estimation
            metric_tapes, tensor_dirs = self._get_tensor_tapes(cost, args, kwargs)
            all_grad_tapes += grad_tapes
            all_metric_tapes += metric_tapes
            all_grad_dirs.append(grad_dirs)
            all_tensor_dirs.append(tensor_dirs)

        if isinstance(cost.device, qml.devices.Device):
            program, config = cost.device.preprocess()

            raw_results = qml.execute(
                all_grad_tapes + all_metric_tapes,
                cost.device,
                None,
                transform_program=program,
                config=config,
            )
        else:
            raw_results = qml.execute(
                all_grad_tapes + all_metric_tapes, cost.device, None
            )  # pragma: no cover
        grads = [
            self._post_process_grad(raw_results[2 * i : 2 * i + 2], all_grad_dirs[i])
            for i in range(self.resamplings)
        ]
        grads = pnp.array(grads)
        metric_tensors = [
            self._post_process_tensor(
                raw_results[2 * self.resamplings + 4 * i : 2 * self.resamplings + 4 * i + 4],
                all_tensor_dirs[i],
            )
            for i in range(self.resamplings)
        ]
        metric_tensors = pnp.array(metric_tensors)
        grad_avg = pnp.mean(grads, axis=0)
        tensor_avg = pnp.mean(metric_tensors, axis=0)

        self._update_tensor(tensor_avg)
        params_next = self._get_next_params(args, grad_avg)

        return params_next[0] if len(params_next) == 1 else params_next

    def _post_process_grad(self, grad_raw_results, grad_dirs):
        r"""Post process the gradient tape results to get the SPSA gradient estimation.

        Args:
            grad_raw_results list[np.array]: list of the two qnode results with input parameters
            perturbed along the ``grad_dirs`` directions
            grad_dirs list[np.array]: list of perturbation arrays along which the SPSA
            gradients are estimated

        Returns:
            list[np.array]: list of gradient arrays. Each gradient array' dimension matches
            the shape of the corresponding input parameter
        """
        loss_plus, loss_minus = grad_raw_results
        return [
            (loss_plus - loss_minus) / (2 * self.finite_diff_step) * grad_dir
            for grad_dir in grad_dirs
        ]

    def _post_process_tensor(self, tensor_raw_results, tensor_dirs):
        r"""Post process the corresponding tape results to get the metric tensor estimation.

        Args:
            tensor_raw_results list[np.array]: list of the four perturbed qnode results to compute
            the estimated metric tensor
            tensor_dirs list[np.array]: list of the two perturbation directions used to compute
            the metric tensor estimation. Perturbations on the different input parameters have
            been concatenated

        Returns:
            pnp.array: estimated Fubini-Study metric tensor
        """
        tensor_raw_results = [result.squeeze() for result in tensor_raw_results]
        # For each element of tensor_raw_results, the first dimension is the measured probability in
        # the computational ket{0} state, which equals the state overlap between the perturbed and
        # unperturbed circuits.
        tensor_finite_diff = (
            tensor_raw_results[0][0]
            - tensor_raw_results[1][0]
            - tensor_raw_results[2][0]
            + tensor_raw_results[3][0]
        )
        return (
            -(
                pnp.tensordot(tensor_dirs[0], tensor_dirs[1], axes=0)
                + pnp.tensordot(tensor_dirs[1], tensor_dirs[0], axes=0)
            )
            * tensor_finite_diff
            / (8 * self.finite_diff_step**2)
        )

    def _get_next_params(self, args, gradient):
        params = []
        non_trainable_indices = []
        for idx, arg in enumerate(args):
            if not getattr(arg, "requires_grad", False):
                non_trainable_indices.append(idx)
                continue
            # if an input parameter is a zero-dimension array, add one dimension to form an array.
            # The returned result will not drop this added dimension
            if arg.shape == ():
                arg = arg.reshape(-1)
            params.append(arg)

        # params_vec and grad_vec group multiple inputs into the same vector to solve the
        # linear equation
        params_vec = pnp.concatenate([param.reshape(-1) for param in params])
        grad_vec = pnp.concatenate([grad.reshape(-1) for grad in gradient])

        new_params_vec = pnp.linalg.solve(
            self.metric_tensor,
            (-self.stepsize * grad_vec + pnp.matmul(self.metric_tensor, params_vec)),
        )
        # reshape single-vector new_params_vec into new_params, to match the input params
        params_split_indices = []
        tmp = 0
        for param in params:
            tmp += param.size
            params_split_indices.append(tmp)
        new_params = pnp.split(new_params_vec, params_split_indices)
        new_params_reshaped = [new_params[i].reshape(params[i].shape) for i in range(len(params))]

        next_args = []
        non_trainable_idx = 0
        trainable_idx = 0

        # merge trainables and non-trainables into the original order
        for idx, arg in enumerate(args):
            if (
                non_trainable_idx < len(non_trainable_indices)
                and idx == non_trainable_indices[non_trainable_idx]
            ):
                next_args.append(arg)
                non_trainable_idx += 1
                continue
            next_args.append(new_params_reshaped[trainable_idx])
            trainable_idx += 1

        return next_args

    def _get_spsa_grad_tapes(self, cost, args, kwargs):
        dirs = []
        args_plus = list(args)
        args_minus = list(args)
        for index, arg in enumerate(args):
            if not getattr(arg, "requires_grad", False):
                continue
            direction = self.rng.choice([-1, 1], size=arg.shape)

            dirs.append(direction)
            args_plus[index] = arg + self.finite_diff_step * direction
            args_minus[index] = arg - self.finite_diff_step * direction

        cost.construct(args_plus, kwargs)
        tape_plus = cost.tape.copy(copy_operations=True)
        cost.construct(args_minus, kwargs)
        tape_minus = cost.tape.copy(copy_operations=True)
        return [tape_plus, tape_minus], dirs

    def _update_tensor(self, tensor_raw):
        def get_tensor_moving_avg(metric_tensor):
            if self.metric_tensor is None:
                self.metric_tensor = pnp.identity(metric_tensor.shape[0])
            return self.k / (self.k + 1) * self.metric_tensor + 1 / (self.k + 1) * metric_tensor

        def regularize_tensor(metric_tensor):
            tensor_reg = pnp.real(sqrtm(pnp.matmul(metric_tensor, metric_tensor)))
            return (tensor_reg + self.reg * pnp.identity(metric_tensor.shape[0])) / (1 + self.reg)

        tensor_avg = get_tensor_moving_avg(tensor_raw)
        tensor_regularized = regularize_tensor(tensor_avg)
        self.metric_tensor = tensor_regularized
        self.k += 1

    def _get_tensor_tapes(self, cost, args, kwargs):
        dir1_list = []
        dir2_list = []
        args_list = [list(args) for _ in range(4)]

        for index, arg in enumerate(args):
            if not getattr(arg, "requires_grad", False):
                continue
            dir1 = self.rng.choice([-1, 1], size=arg.shape)
            dir2 = self.rng.choice([-1, 1], size=arg.shape)

            dir1_list.append(dir1.reshape(-1))
            dir2_list.append(dir2.reshape(-1))

            args_list[0][index] = arg + self.finite_diff_step * (dir1 + dir2)
            args_list[1][index] = arg + self.finite_diff_step * dir1
            args_list[2][index] = arg + self.finite_diff_step * (-dir1 + dir2)
            args_list[3][index] = arg - self.finite_diff_step * dir1
        dir_vecs = (pnp.concatenate(dir1_list), pnp.concatenate(dir2_list))
        tapes = [
            self._get_overlap_tape(cost, args, args_finite_diff, kwargs)
            for args_finite_diff in args_list
        ]

        return tapes, dir_vecs

    def _get_overlap_tape(self, cost, args1, args2, kwargs):
        # the returned tape effectively measure the fidelity between the two parametrized circuits
        # with input args1 and args2. The measurement results of the tape are an array of probabilities
        # in the computational basis. The first element of the array represents the probability in
        # \ket{0}, which equals the fidelity.
        op_forward = self._get_operations(cost, args1, kwargs)
        op_inv = self._get_operations(cost, args2, kwargs)

        new_ops = op_forward + [qml.adjoint(op) for op in reversed(op_inv)]
        return qml.tape.QuantumScript(new_ops, [qml.probs(wires=cost.tape.wires.labels)])

    @staticmethod
    def _get_operations(cost, args, kwargs):
        cost.construct(args, kwargs)
        return cost.tape.operations

    def _apply_blocking(self, cost, args, kwargs, params_next):
        cost.construct(args, kwargs)
        tape_loss_curr = cost.tape.copy(copy_operations=True)

        if not isinstance(params_next, list):
            params_next = [params_next]

        cost.construct(params_next, kwargs)
        tape_loss_next = cost.tape.copy(copy_operations=True)

        if isinstance(cost.device, qml.devices.Device):
            program, _ = cost.device.preprocess()

            loss_curr, loss_next = qml.execute(
                [tape_loss_curr, tape_loss_next], cost.device, None, transform_program=program
            )

        else:
            loss_curr, loss_next = qml.execute([tape_loss_curr, tape_loss_next], cost.device, None)

        # self.k has been updated earlier
        ind = (self.k - 2) % self.last_n_steps.size
        self.last_n_steps[ind] = loss_curr

        tol = (
            2 * self.last_n_steps.std()
            if self.k > self.last_n_steps.size
            else 2 * self.last_n_steps[: self.k - 1].std()
        )

        if loss_curr + tol < loss_next:
            params_next = args

        if len(params_next) == 1:
            return params_next[0], loss_curr
        return params_next, loss_curr
