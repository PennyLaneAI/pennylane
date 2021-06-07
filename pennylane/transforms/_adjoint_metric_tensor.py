import pennylane as qml
from pennylane import numpy as np
import numpy as onp

from autograd.extend import defvjp
from autograd.numpy.numpy_vjps import match_complex
from autograd.numpy import numpy_wrapper as anp

def vdot_adjoint_0(B, G, A_meta, B_meta):
    A_shape, A_ndim, *_ = A_meta
    flat_dim = anp.prod(A_shape)
    G = anp.conj(anp.reshape(G, (*G.shape[:-A_ndim], flat_dim)))
    B = B.reshape(flat_dim)
    return anp.dot(G, B)

def vdot_adjoint_1(A, G, A_meta, B_meta):
    B_shape, B_ndim, *_ = B_meta
    flat_dim = anp.prod(B_shape)
    G = anp.reshape(G, (*G.shape[:-B_ndim], flat_dim))
    A = anp.conj(A.reshape(flat_dim))
    return anp.dot(G, A)

def vdot_vjp_0(ans, A, B):
    A_meta, B_meta = anp.metadata(A), anp.metadata(B)
    return lambda g: match_complex(A, vdot_adjoint_0(B, g, A_meta, B_meta))

def vdot_vjp_1(ans, A, B):
    A_meta, B_meta = anp.metadata(A), anp.metadata(B)
    return lambda g: match_complex(B, vdot_adjoint_1(A, g, A_meta, B_meta))

defvjp(anp.vdot, vdot_vjp_0, vdot_vjp_1)

def _get_generator(op):
    """Reads out the generator and prefactor of an operation and converts
    to matrix if necessary.

    Args:
        op (:class:`~.Operation`): Operation to obtain the generator of.
    Returns:
        array[float]: Generator matrix
        float: Prefactor of the generator
    """

    generator, prefactor = op.generator
    if not isinstance(generator, np.ndarray):
        generator = generator.matrix
    if op.inverse:
        generator = generator.conj().T
        prefactor *= -1

    return generator, prefactor

def _adjoint_metric_tensor(tape, device, wrt):
    """Implements the adjoint method outlined in
    `Jones <https://arxiv.org/abs/2011.02991>`__ to compute the metric tensor.

    A mixture of a main forward pass and intermediate partial backwards passes is
    used to evaluate the metric tensor in O(num_params^2) operations, using 4 state
    vectors.

    .. note::
        The adjoint metric tensor method has the following restrictions:

        * As it requires knowledge of the statevector, only statevector simulator
          devices can be used.

        * We assume the circuit to be composed of unitary gates only and rely
          on the ``generator`` property of the gates to be implemented.
          Note also that this makes the metric tensor real-valued.

    Args:
        tape (.QuantumTape): circuit that the function computes the metric tensor of

    Returns:
        array: the metric tensor of the tape with respect to its trainable parameters.
        Dimensions are ``(len(trainable_params), len(trainable_params))``.

    Raises:
        QuantumFunctionError: if the input tape contains a multi-parameter 
            operation aside from :class:`~.Rot` or an operation without 
            ``generator`` attribute.
    """

    def _apply_any_operation(state, op):
        if isinstance(op, qml.QubitStateVector):
            device._apply_state_vector(op.parameters[0], op.wires)
            return device._state
        elif isinstance(op, qml.BasisState):
            device._apply_basis_state(op.parameters[0], op.wires)
            return device._state
        else:
            return device._apply_operation(state, op)

    if wrt is not None:
        tape.trainable_params = set(wrt)
    # generate and extract initial state
    psi = device._create_basis_state(0)  # pylint: disable=protected-access
    dim = 2**device.num_wires

    # initialize tensor components (which all will be real-valued)
    L = []
    L_diag = []
    T = []

    # preprocessing: check support of adjoint method for all operations
    # and handle special case of qml.Rot
    expanded_ops = []
    param_number = 0
    for op in tape.operations:
        if op.num_params > 1:
            if isinstance(op, qml.Rot):
                op_params = [-p for p in op.parameters[::-1]] if op.inverse else op.parameters
                ops = op.decomposition(*op_params, wires=op.wires)
                expanded_ops.extend(ops)
                param_number += 3
                continue
            else:
                raise qml.QuantumFunctionError(
                    f"The {op.name} operation is not supported using "
                    'the "adjoint" metric tensor method.'
                )
        if all([
                op.grad_method is not None,
                param_number in tape.trainable_params,
                not hasattr(op, "generator") or op.generator[0] is None
                ]):
            raise qml.QuantumFunctionError(
                "The adjoint metric tensor method requires operations to"
                f"have a 'generator' attribute but {op.name} does not have one."
            )
        param_number += 1
        expanded_ops.append(op)

    # preprocessing: divide all operations into trainable operations and blocks
    # of untrainable operations after each trainable one
    trainable_operations = []
    after_trainable_operations = {}
    param_number = -1
    # the first set of non-trainable ops are the ops "after the -1st" trainable op
    train_param_number = -1
    after_prev_train_op = []
    for i, op in enumerate(expanded_ops):
        # check whether we recognize the gate as parametrized
        if op.grad_method is not None:
            param_number += 1
            # check whether the parameter is among the ones we are training
            if param_number in tape.trainable_params:
                trainable_operations.append(op)
                after_trainable_operations[train_param_number] = after_prev_train_op
                train_param_number += 1
                after_prev_train_op = []
                continue
        after_prev_train_op.append(op)
    # store operations after last trainable op
    after_trainable_operations[train_param_number] = after_prev_train_op

    for op in after_trainable_operations[-1]:
        psi = _apply_any_operation(psi, op)

    num_params = len(tape.trainable_params)
    for j, outer_op in enumerate(trainable_operations):
        sub_L = []
        lam = np.array(psi, copy=True)
        phi = np.array(psi, copy=True)
        generator1, prefactor1 = _get_generator(outer_op)

        # the state vector phi is missing a factor of 1j * prefactor1
        phi = device._apply_unitary(phi, generator1, outer_op.wires)
        phi_real = np.real(phi).reshape(dim)
        phi_imag = np.imag(phi).reshape(dim)
        L_diag.append(prefactor1**2 * (np.dot(phi_real, phi_real) + np.dot(phi_imag, phi_imag)))

        lam_real = np.real(lam).reshape(dim)
        lam_imag = np.imag(lam).reshape(dim)
        # this entry is missing a factor of 1j
        T.append(prefactor1 * (np.dot(lam_real, phi_real) + np.dot(lam_imag, phi_imag)))

        for i in range(j-1, -1, -1):
            # after first iteration of inner loop: apply U_{i+1}^\dagger
            if i<j-1:
                trainable_operations[i+1].inv()
                phi = device._apply_operation(phi, trainable_operations[i+1])
                trainable_operations[i+1].inv()
            # apply V_{i}^\dagger
            for op in after_trainable_operations[i][::-1]:
                op.inv()
                phi = _apply_any_operation(phi, op)
                lam = _apply_any_operation(lam, op)
                op.inv()
            mu = np.array(lam, copy=True)
            inner_op = trainable_operations[i]
            # extract and apply G_i
            generator2, prefactor2 = _get_generator(inner_op)
            # this state vector is missing a factor of 1j * prefactor2
            mu = device._apply_unitary(mu, generator2, inner_op.wires)
            phi_real = np.real(phi).reshape(dim)
            phi_imag = np.imag(phi).reshape(dim)
            mu_real = np.real(mu).reshape(dim)
            mu_imag = np.imag(mu).reshape(dim)
            # this entry is missing a factor of 1j * (-1j) = 1, i.e. none
            sub_L.append(prefactor1 * prefactor2 * (np.dot(mu_real, phi_real) + np.dot(mu_imag, phi_imag)))
            # apply U_i^\dagger
            inner_op.inv()
            lam = device._apply_operation(lam, inner_op)
            inner_op.inv()
        L.extend(sub_L[::-1])
        L.extend([0.0]*(num_params-j))

        # apply U_j
        psi = device._apply_operation(psi, outer_op)
        # apply V_j
        for op in after_trainable_operations[j]:
            psi = _apply_any_operation(psi, op)

    # postprocessing: combine L, L_diag and T into the metric tensor.
    # We require outer(conj(T), T) here, but as we skipped the factor 1j above,
    # the stored T is real-valued. Thus we have -1j*1j*outer(T, T) = outer(T, T)
    L = np.reshape(L, (num_params, num_params))
    metric_tensor = L + L.T + np.eye(num_params)*L_diag - np.outer(T, T)

    return metric_tensor

