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
"""Contains a transform that computes the frequency spectrum of a quantum
circuit."""
from itertools import product, combinations
from functools import wraps
from collections import OrderedDict
import warnings
import numpy as np
import pennylane as qml
from inspect import signature


def _get_spectrum(op, decimals=8):
    r"""Extract the frequencies contributed by an input-encoding gate to the
    overall Fourier representation of a quantum circuit.

    If :math:`G` is the generator of the input-encoding gate :math:`\exp(-i x G)`,
    the frequencies are the differences between any two of :math:`G`'s eigenvalues.
    We only compute non-negative frequencies in this subroutine.

    Args:
        op (~pennylane.operation.Operation): :class:`~.pennylane.Operation` to extract
            the frequencies for
        decimals (int): Number of decimal places to round the frequencies to

    Returns:
        set[float]: non-negative frequencies contributed by this input-encoding gate
    """
    no_generator = False
    if hasattr(op, "generator"):
        g, coeff = op.generator

        if isinstance(g, np.ndarray):
            matrix = g
        elif hasattr(g, "matrix"):
            matrix = g.matrix
        else:
            no_generator = True
    else:
        no_generator = True

    if no_generator:
        raise ValueError(f"Generator of operation {op} is not defined.")

    matrix = coeff * matrix
    # todo: use qml.math.linalg once it is tested properly
    evals = np.linalg.eigvalsh(matrix)

    # compute all unique positive differences of eigenvalues, then add 0
    _spectrum = set(
        np.round(np.abs([x[1] - x[0] for x in combinations(evals, 2)]), decimals=decimals)
    )
    _spectrum |= {0}

    return _spectrum


def _join_spectra(spec1, spec2):
    r"""Join two sets of frequencies that belong to the same input.

    Since :math:`\exp(i a x)\exp(i b x) = \exp(i (a+b) x)`, the spectra of two gates
    encoding the same :math:`x` are joined by computing the set of sums and absolute
    values of differences of their elements.
    We only compute non-negative frequencies in this subroutine and assume the inputs
    to be non-negative frequencies as well.

    Args:
        spec1 (set[float]): first spectrum
        spec2 (set[float]): second spectrum
    Returns:
        set[float]: joined spectrum
    """
    if spec1 in ({0}, {}):
        return spec2
    if spec2 in ({0}, {}):
        return spec1

    sums = {s1 + s2 for s1 in spec1 for s2 in spec2}
    diffs = {np.abs(s1 - s2) for s1 in spec1 for s2 in spec2}

    return sums.union(diffs)

def _get_random_args(args, interface, num, seed):
    r"""Generate random arguments of the same shapes as provided args.

    Args:
        args (tuple): Original input arguments
        interface (str): Interface of the QNode into which the arguments will be fed
        num (int): Number of random argument sets to generate
    Returns:
        list[tuple]: List of length ``num`` with each entry being a random instance
        of arguments like ``args``.
    """
    if interface == "tf":
        import tensorflow as tf
        tf.random.set_seed(seed)
        rnd_args = []
        for _ in range(num):
            _args = (tf.random.uniform(_arg.shape)*2*np.pi-np.pi for _arg in args)
            _args = (
                tf.Variable(_arg) if isinstance(arg, tf.Variable) else _arg
                for _arg, arg in zip(_args, args)
            )
            rnd_args.append(_args)
    elif interface == "torch":
        import torch
        torch.random.manual_seed(seed)
        rnd_args = [
            tuple(torch.rand(arg.shape)*2*np.pi-np.pi for arg in args)
            for _ in range(num)
        ]
    else:
        np.random.seed(seed)
        rnd_args = [
            tuple(np.random.random(np.shape(arg))*2*np.pi-np.pi for arg in args)
            for _ in range(num)
        ]

    return rnd_args

def _get_and_validate_classical_jacobian(qnode, argnum, args, kwargs, num_pos=1):
    r"""Check classical preprocessing of a QNode to be linear and return its Jacobian.

    Args:
        qnode (pennylane.QNode): a quantum node of which to validate the preprocessing
        argnum (list[int]): the indices of the arguments with respect to which the Jacobian
            is computed; passed to `~pennylane.transforms.classical_jacobian`
        args (tuple): QNode arguments; the input parameters are one of four positions at which
            the Jacobian is computed, and the QNode arguments are left at these values
        kwargs (dict): QNode keyword arguments
        num_pos (int): Number of additional random positions at which to evaluate the
            Jacobian and test that it is constant

    Returns:
        (tuple[array]): Jacobian of the classical preprocessing (at QNode arguments args).

    The output of the `~pennylane.QNode` is only a Fourier series in the encoded :math:`x_i`
    if the processing of the QNode parameters into gate parameters is linear.
    This method asserts this linearity by computing the Jacobian of the processing at
    multiple positions and checking that it is constant.
    """
    try:
        # Get random input arguments
        rnd_args = _get_random_args(args, qnode.interface, num_pos, seed=291)
        # Evaluate the classical Jacobian at multiple input args.
        jacs = [
            qml.transforms.classical_jacobian(qnode, argnum=argnum)(*_args, **kwargs)
            for _args in rnd_args
        ]
    except Exception as e:
        raise ValueError("Could not compute Jacobian of the classical preprocessing.") from e

    # Check that the Jacobian is constant
    if not all(
        (all((np.allclose(jacs[0][i], jac[i]) for jac in jacs[1:])) for i in range(len(jacs[0])))
    ):
        raise ValueError(
            "The Jacobian of the classical preprocessing in the provided QNode is not constant; "
            "only linear classical preprocessing is supported."
        )

    # Note that jacs is a list of tuples of arrays
    return jacs[0]

def _process_ids(encoding_args, argnum, qnode):
    sig_pars = signature(qnode.func).parameters
    arg_names = [name for name, par in sig_pars.items() if par.default is par.empty]

    if encoding_args is None:
        if argnum is None:
            encoding_args = OrderedDict((name, ...) for name in arg_names)
            argnum = list(range(len(arg_names)))
        elif np.isscalar(argnum):
            encoding_args = OrderedDict({arg_names[argnum]: ...})
            argnum = [argnum]
        else:
            encoding_args = OrderedDict((arg_names[num], ...) for num in argnum)
            argnum = argnum
    else:
        requested_names = set(encoding_args)
        if not all(name in arg_names for name in requested_names):
            raise ValueError(
                f"Not all names in {requested_names} are known. "
                f"Known arguments: {arg_names}"
            )
        # Selection of requested argument names from sorted names
        encoding_args = OrderedDict(
            (name, encoding_args[name]) for name in arg_names if name in requested_names
        )
        argnum = [arg_names.index(name) for name in encoding_args]
    return encoding_args, argnum


def spectrum(qnode, encoding_args=None, argnum=None, decimals=5):

    encoding_args, argnum = _process_ids(encoding_args, argnum, qnode)
    atol = 10 ** (-decimals) if decimals is not None else 1e-10
    # A map between Jacobians (contiguous) and arg names (may be discontiguous)
    arg_name_map = dict(enumerate(encoding_args))

    @wraps(qnode)
    def wrapper(*args, **kwargs):
        # Compute classical Jacobian and assert preprocessing is linear
        class_jacs = _get_and_validate_classical_jacobian(qnode, argnum, args, kwargs)

        spectra = {}
        par_info = qnode.qtape._par_info
        for jac_idx, class_jac in enumerate(class_jacs):
            arg_name = arg_name_map[jac_idx]
            if encoding_args[arg_name] is Ellipsis:
                encoding_args[arg_name] = product(*(range(sh) for sh in class_jac.shape[1:]))
            requested_par_ids = set(encoding_args[arg_name])
            _spectra = {par_idx: {0} for par_idx in requested_par_ids}

            for op_idx, jac_of_op in enumerate(np.round(class_jac, decimals=decimals)):
                op = par_info[op_idx]["op"]
                # Find parameters that where requested and feed into the operation
                if len(class_jac.shape) == 1:
                    # Scalar argument, only axis of Jacobian is for gates
                    if np.isclose(jac_of_op, 0.):
                        continue
                    jac_of_op = {(): jac_of_op}
                    par_ids = {()}
                else:
                    par_ids = zip(*[map(int, _ids) for _ids in np.where(jac_of_op)])
                    par_ids = set(par_ids).intersection(requested_par_ids)
                    if len(par_ids) == 0:
                        continue
                # Multi-parameter gates are not supported
                if len(op.parameters) != 1:
                    raise ValueError(
                        "Can only consider one-parameter gates as data-encoding gates; "
                        f"got {op.name}."
                    )
                spec = _get_spectrum(op, decimals=decimals)

                for par_idx in par_ids:
                    scale = float(jac_of_op[par_idx])
                    scaled_spec = [scale * f for f in spec]
                    _spectra[par_idx] = _join_spectra(_spectra[par_idx], scaled_spec)

            # Construct the sorted spectrum also containing negative frequencies
            for idx, spec in _spectra.items():
                spec = sorted(spec)
                _spectra[idx] = [-freq for freq in spec[:0:-1]] + spec
            spectra[arg_name] = _spectra

        return spectra

    return wrapper
