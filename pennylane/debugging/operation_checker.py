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
r"""
This file contains the OperationChecker debugging and developing tool.
"""
import inspect
from functools import partial

import scipy.linalg as la

import pennylane as qml
from pennylane import numpy as np
from pennylane.operation import (
    MatrixUndefinedError,
    SparseMatrixUndefinedError,
    GeneratorUndefinedError,
    DiagGatesUndefinedError,
    EigvalsUndefinedError,
    TermsUndefinedError,
    DecompositionUndefinedError,
    AnyWires,
)

try:
    import jax

    has_jax = True
except ModuleNotFoundError:
    has_jax = False
try:
    import tensorflow as tf

    has_tf = True
except ModuleNotFoundError:
    has_tf = False
try:
    import torch

    has_torch = True
except ModuleNotFoundError:
    has_torch = False

_colors = {
    "error": "91",  # red
    "fatal_error": "91",  # red
    "hint": "93",  # yellow
    "comment": 94,  # blue
    "pass": "92",  # green
}

verbosity_levels = {"fatal_error": 0, "error": 1, "hint": 2, "comment": 3, "pass": 4}
levels_verbosity = {val: key for key, val in verbosity_levels.items()}

_default_methods_to_check = [
    ("compute_eigvals", EigvalsUndefinedError, False),
    ("compute_matrix", MatrixUndefinedError, False),
    ("compute_sparse_matrix", SparseMatrixUndefinedError, False),
    ("compute_terms", TermsUndefinedError, False),
    ("compute_decomposition", DecompositionUndefinedError, True),
    ("compute_diagonalizing_gates", DiagGatesUndefinedError, True),
]


def equal_up_to_phase(mat1, mat2, atol=1e-10):
    r"""Check whether two matrices are equal up to a scalar
    prefactor of the form :math:`\exp(i\phi)`.

    Args:
        mat1 (array_like): First matrix to check for equality
        mat2 (array_like): Second matrix to check for equality
        atol (float): Absolute tolerance for the check for equality

    Return:
        bool: Whether the two input matrices are equal up to a scalar
        phase prefactor.
    """
    # Check whether the matrices are equal
    if np.allclose(mat1, mat2, atol=atol, rtol=0.0):
        return True

    # Compute the potential scalar prefactor from the first nonzero entry of mat2
    ids = np.where(np.round(mat2, 10))
    idx = (ids[0][0], ids[1][0])
    phase = mat1[idx] / mat2[idx]

    # Return whether the matrices are equal, accounting for the potential scalar prefactor
    return np.isclose(np.abs(phase), 1.0, atol=atol) and np.allclose(
        mat1, mat2 * phase, atol=atol, rtol=0.0
    )


def is_diagonal(matrix, atol=1e-10):
    r"""Check whether a matrix is a diagonal matrix

    Args:
        matrix (array_like): Matrix to check

    Returns:
        bool: Whether the input matrix is a diagonal matrix
    """
    # Extract the diagonal, subtract it from the input, and check whether the result is 0.
    off_diagonal = matrix - np.diag(np.diag(matrix))
    return np.allclose(off_diagonal, np.zeros_like(matrix), atol=atol, rtol=0.0)


def torch_jacobian(fn):
    """Functional jacobian in the torch interface with the same
    syntax as qml.jacobian. Note that this is not performant!"""
    return lambda *args: torch.autograd.functional.jacobian(fn, args)


def tf_jacobian(fn):
    """Functional jacobian in the tensorflow interface with the same
    syntax as qml.jacobian. Note that this is not performant!"""

    def jac_fn(*args):
        with tf.GradientTape() as tape:
            out = fn(*args)
        return tuple(0.0 if val is None else val.numpy() for val in tape.jacobian(out, args))

    return jac_fn


def complex_jacobian(fn, jacobian_fn=qml.jacobian):
    r"""Compute the Jacobian of a complex-valued function with real-valued inputs.

    Args:
        fn (callable): Function to differentiate

    Returns:
        callable: Jacobian function with the same input signature as the input function.
        It returns the linear combination of the real and imaginary part of the function
        output, with coefficient :math:`1` and :math:`i`.

    .. warning::

        The derivative of a complex-valued function can be defined in multiple
        ways, so that the returned function may differ from other implementations
        of functions :math:`f:\mathbb{R}\mapsto\mathbb{C}^k`.

    """
    r_jac = jacobian_fn(lambda *args, **kwargs: qml.math.real(fn(*args, **kwargs)))
    i_jac = jacobian_fn(lambda *args, **kwargs: qml.math.imag(fn(*args, **kwargs)))

    def complex_jac(*args, **kwargs):
        rjac = r_jac(*args, **kwargs)
        ijac = i_jac(*args, **kwargs)
        if isinstance(rjac, tuple):
            return tuple(r + 1j * i for r, i in zip(rjac, ijac))

        return rjac + 1j * ijac

    return complex_jac


def recipe_yields_commutator(recipe, op, par, wires):
    r"""Check whether a gradient recipe produces the commutator as required to
    produce the correct derivative.

    .. warning::

        This test only is implemented for single-parameter gates.

    Args:
        recipe (array_like): Gradient recipe with shape ``(3, M)``, where the rows
            correspond to the coefficient, multipliers and shifts, respectively.
        op (type): Operation subclass to be checked.
        par (array_like): Parameter position at which to check the operation.
            As the test is only implemented for single-parameter operations,
            the shape of ``par`` is expected to be ``(1,)``.
        wires (.Wires): Wires with which to instantiate the checked operation.

    Returns:
        bool: Whether the provided ``recipe`` produces the commutator belonging
        to the checked ``op`` at the given parameter position, and thus will produce
        the correct derivative.

    For any single-parameter gate of the form :math:`U(x)=\exp(ixG)` with some Hermitian
    generator :math:`G`, we can consider a cost function arising from measuring a
    Hamiltonian :math:`H` after applying the gate:

    .. math::

        E(x) = \langle \psi | U(x)^\dagger H U(x) | \psi \rangle

    The derivative of :math:`E` then is given by:

    .. math::

        \partial_x E(x) = i \langle \psi | U(x)^\dagger [H, G] U(x) | \psi \rangle

    This means: if a gradient recipe applied to the operator-valued function

    .. math::

        O(x) = U(x)^\dagger H U(x)

    produces the commutator :math:`i[H, G]`, it will produce the correct derivative when
    applied to the function :math:`E(x)`.

    In this function, we check that both :math:`i[H, G]` and :math:`\partial_H i[H, G]`
    (in the sense of a component-wise derivative) are correct for some random :math:`H`,
    which is very likely to be a sufficient test.
    """

    coeffs, multipliers, shifts = recipe
    shifted_parameters = [m * par[0] + s for m, s in zip(multipliers, shifts)]
    shifted_matrices = [qml.matrix(op(p, wires=wires)) for p in shifted_parameters]

    def apply_recipe(H):
        out = qml.math.zeros_like(H)
        for c, mat in zip(coeffs, shifted_matrices):
            out = out + c * mat.T.conj() @ H @ mat
        return out

    d_apply_recipe = complex_jacobian(apply_recipe)

    mat_fn = lambda p: qml.matrix(op(p, wires=wires))
    d_mat_fn = complex_jacobian(mat_fn)
    orig_mat = mat_fn(par[0])
    d_orig_mat = d_mat_fn(par[0])
    commutator_fn = (
        lambda H: d_orig_mat.conj().T @ H @ orig_mat + orig_mat.conj().T @ H @ d_orig_mat
    )
    d_commutator_fn = complex_jacobian(commutator_fn)

    dim = 2 ** len(wires)
    H = (
        np.random.random((dim, dim), requires_grad=True) * 2
        + np.random.random((dim, dim), requires_grad=True) * 2j
    )
    H += H.T.conj()
    return np.allclose(d_apply_recipe(H), d_commutator_fn(H))


def wrap_op_method(op, method, expected_exc):
    r"""Wrap a method of an operation with a try-except clause, allowing for an expected
    exception and catching (and returning) other exceptions.

    Args:
        op (type or .operation.Operation): Operation that has the method to be wrapped
        method (str): Name of the method to be wrapped
        expected_exc (type): Exception type to ignore

    Returns:
        callable: The wrapped method of the operation.

    The object returned by the returned callable differs, depending on the scenario:

      - If the method succeeds, the return value of the method is returned,
      - If the ``expected_exc`` is raised, ``None`` is returned,
      - If another exception is raised, it is returned (but not raised).
    """
    _method = getattr(op, method)

    def wrapped_method(*args, **kwargs):
        r"""Wrapped operation method that tolerates an expected exception
        and catches (and returns) all other exceptions."""
        try:
            return _method(*args, **kwargs)
        except expected_exc:
            return None
        except Exception as e:  # pylint: disable=broad-except
            return e

    return wrapped_method


def matrix_from_hardcoded_matrix(op, par, wires):
    r"""Get the matrix of an operation, using ``get_matrix``.

    Args:
        op (type): Operation type to obtain the matrix for
        par (array_like): Parameters of the operation
        wires (.wires.Wires): Wires of the operation

    Returns:
        object: Matrix of the operation if it is defined and no problem occured
        with ``get_matrix``. ``None`` if no matrix is defined via ``get_matrix``
        or ``Exception`` if an error occured.
    """
    instance = op(*par, wires=wires)
    return wrap_op_method(instance, "get_matrix", MatrixUndefinedError)()


def matrix_from_sparse_matrix(op, par, wires):
    r"""Get the matrix of an operation, using ``sparse_matrix``.

    Args:
        op (type): Operation type to obtain the matrix for
        par (array_like): Parameters of the operation
        wires (.wires.Wires): Wires of the operation

    Returns:
        object: Matrix of the operation if it is defined and no problem occured
        with ``sparse_matrix``. ``None`` if no matrix is defined via ``sparse_matrix``
        or ``Exception`` if an error occured.
    """
    instance = op(*par, wires=wires)
    smat = wrap_op_method(instance, "sparse_matrix", SparseMatrixUndefinedError)()
    if smat is None or isinstance(smat, Exception):
        return smat
    return smat.toarray()


def matrix_from_terms(op, par, wires):
    r"""Get the matrix of an operation, using its ``terms``.

    Args:
        op (type): Operation type to obtain the matrix for
        par (array_like): Parameters of the operation
        wires (.wires.Wires): Wires of the operation

    Returns:
        object: Matrix of the operation if it is defined and no problem occured
        with ``terms``. ``None`` if no terms are defined via ``terms``
        or ``Exception`` if an error occured.
    """
    instance = op(*par, wires=wires)
    terms = wrap_op_method(instance, "terms", TermsUndefinedError)()
    if terms is None or isinstance(terms, Exception):
        return terms

    return qml.matrix(qml.Hamiltonian(*terms))


def matrix_from_decomposition(op, par, wires):
    r"""Get the matrix of an operation, using its ``decomposition``.

    Args:
        op (type): Operation type to obtain the matrix for
        par (array_like): Parameters of the operation
        wires (.wires.Wires): Wires of the operation

    Returns:
        object: Matrix of the operation if it is defined and no problem occured
        with ``expand``. ``None`` if no decomposition is defined via ``expand``
        or ``Exception`` if an error occured.
    """
    instance = op(*par, wires=wires)
    tape = wrap_op_method(instance, "expand", DecompositionUndefinedError)()
    return qml.matrix(tape, wire_order=wires) if isinstance(tape, qml.tape.QuantumTape) else tape


def matrix_from_single_qubit_rot_angles(op, par, wires):
    r"""Get the matrix of an operation, using its ``single_qubit_rot_angles``.

    Args:
        op (type): Operation type to obtain the matrix for
        par (array_like): Parameters of the operation
        wires (.wires.Wires): Wires of the operation

    Returns:
        object: Matrix of the operation if it is defined and no problem occured
        with ``single_qubit_rot_angles``. ``None`` if no rotation angles are
        defined via ``single_qubit_rot_angles`` or ``Exception`` if an error occured.
    """
    instance = op(*par, wires=wires)
    try:
        angles = instance.single_qubit_rot_angles()
    except (AttributeError, NotImplementedError):
        return None
    with qml.tape.QuantumTape() as tape:
        qml.RZ(angles[0], wires=wires)
        qml.RY(angles[1], wires=wires)
        qml.RZ(angles[2], wires=wires)
    return qml.matrix(tape)


def matrix_from_generator(op, par, wires):
    r"""Get the matrix of an operation, using its ``generator``.

    .. warning::

        Only carries out the test for single-parameter gates.

    Args:
        op (type): Operation type to obtain the matrix for
        par (array_like): Parameters of the operation
        wires (.wires.Wires): Wires of the operation

    Returns:
        object: Matrix of the operation if it is defined and no problem occured
        with ``generator``. ``None`` if no generator is defined via ``generator``
        or ``Exception`` if an error occured.
    """
    if len(par) != 1:
        return None
    instance = op(*par, wires=wires)
    try:
        gen = qml.generator(instance, "observable")
    except GeneratorUndefinedError:
        return None
    mat = qml.matrix(gen)

    return la.expm(1j * par[0] * mat)


decomposition_methods = [
    matrix_from_single_qubit_rot_angles,
    matrix_from_hardcoded_matrix,
    matrix_from_sparse_matrix,
    matrix_from_terms,
    matrix_from_decomposition,
    matrix_from_generator,
]


class CheckerError(Exception):
    """An internal error raised in OperationChecker used to mark specific exceptions."""


class OperationChecker:
    r"""Check one or multiple operation subclasses to define all required properties,
    be well-defined, and have consistent properties.

    Args:
        verbosity (str): How much output to print during execution (also see below):

            - ``"pass"``: Print all errors, hints, comments and status reports;

            - ``"comment"``: Like ``"pass"`` but without status reports;

            - ``"hint"``: Only print errors and hints;

            - ``"error"``: Only print errors.

        max_num_params (int): Largest number of parameters to check for operations
            that do not provide a fixed number of parameters via ``num_params`` themselves.
        print_fn (callable): Function used to store or print output. Must take a single string
            as input.
        print_color (bool): Whether or not to use colors in the terminal and returned outputs.
        tol (float): Numeric (absolute) tolerance when comparing matrices.

    The categorization of test results and of the associated messages is as follows:

    - ``"pass"``: Status reports e.g., after a successfully completed run.

    - ``"comment"``: Comments regarding certain properties of the tested operation(s).
      These does not require any action to change the operation but is used to raise awareness
      for behaviour that might be unexpected or differing from common operations.

    - ``"hint"``: Remarks similar to warnings that indicate concrete hints to change the checked
      operation(s). The recommended changes are expected to improve the code quality, performance
      or consistency with other operations.

    - ``"error"``: Problems with the checked operation(s) that require changes. These problems
      might be in the core of the ``Operation``, preventing instantiation, or in a specific
      method or property that is rendered unusable or inconsistent by the problem.


    **Example**

    Once instantiated, the ``OperationChecker`` can be called on an operation instance or class
    with the following signature:

    Args:

      - op (type or .operation.Operation): Operation(s) to check, either provided as class
        or as instance.

      - parameters (Sequence[int or float]): Parameters with which the operation(s) is/are
        expected to work. Ignored if ``op`` is not a type but a class instance. If not
        provided, random parameters are used.

      - wires (.wires.Wires): Wires with which the operation(s) is/are expected to work.
        Ignored if ``op`` is not a type but a class instance. If not provided, consecutive
        integer-labelled wires are used.

      - seed (int): Seed for random generation of parameters.


    Returns:

      - str: The result status of the operation check, one of ``"error"``, ``"hint"``,
        ``"comment"``, and ``"pass"``.
      - str: A copy of the text printed via the ``print_fn`` during the check.

    Let's consider a custom operation, which is a copy of ``qml.RX`` but with its
    argument rounded to 2 decimal places:

    .. code-block ::

        class MyRX(qml.operation.Operation):

            num_wires = 1

            def __init__(self, theta, wires, do_queue=True, id=None):
                theta = qml.math.round(theta, 2)
                super().__init__(theta, wires=wires, do_queue=do_queue, id=id)

            @staticmethod
            def compute_matrix(theta):
                theta = qml.math.round(theta, 2)
                return qml.math.array([
                    [qml.math.cos(theta / 2), -1j * qml.math.sin(theta / 2)],
                    [-1j * qml.math.sin(theta / 2), qml.math.cos(theta / 2)],
                ])

    Now we can run the OperationChecker on this operation class:

    >>> checker = qml.debugging.OperationChecker()
    >>> result, output = checker(MyRX)
    Checking operation MyRX for consistency.
    = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    \x1b[93mInstantiating MyRX only succeeded when using 1 parameter(s).
    Consider specifying the number of parameters by setting op.num_params.\x1b[0m

    As we can see, an improvement to ``MyRX`` is easily possible by providing it with
    the property ``num_params``. The return values ``result`` and ``output`` contain
    the status of the check and the printed output, respectively:

    >>> print(result)
    hint

    .. UsageDetails::

        **Filtering messages**

        By specifying a ``verbosity`` level, less important messages may be suppressed.
        For example, if only errors and hints should be displayed, but commentary that
        does not require any action should be suppressed, set ``verbosity="hint"``.

        **Retrieving previous check results**

        An instance of ``OperationChecker`` is stateful, allowing us to retrieve the results
        of checks that were performed earlier:

        >>> checker = qml.debugging.OperationChecker(verbosity="error")
        >>> for op in [qml.RX, qml.RY, qml.IsingZZ]:
        >>>     checker(op)
        >>> print(checker.results[qml.RX])
        pass

        >>> checker.outputs[qml.RY]
        ''

        **Using a custom print function**

        By providing a custom function via ``print_fn``, we e.g. can store
        the output to a file:

        .. code-block::

            import sys

            def print_to_file(string):
                '''Print string to fixed file'''
                original_stdout = sys.stdout

                with open('check_output.txt', 'a') as f:
                    sys.stdout = f
                    print(string)
                    sys.stdout = original_stdout

        >>> checker = qml.debugging.OperationChecker(print_fn=print_to_file)
        >>> checker(qml.RX)
        >>> with open("check_output.txt", "r") as f:
        ...     for line in f.readlines():
        ...         print(line)
        Checking operation RX for consistency.
        = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
        No problems have been found with the operation RX.
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self, verbosity="pass", max_num_params=10, print_fn=print, print_color=True, tol=1e-5
    ):
        # pylint: disable=too-many-arguments
        self._verbosity = {
            key for key, val in verbosity_levels.items() if val <= verbosity_levels[verbosity]
        }
        self.max_num_params = max_num_params
        self.print_fn = print_fn
        self.print_color = print_color
        self.tol = tol
        self.results = {}
        self.outputs = {}
        self.tmp = self.seed = None

    def __call__(self, op, parameters=None, wires=None, seed=None):
        r"""Call the OperationChecker on one or multiple operations."""
        self.seed = seed
        # Initialize result status and output for this operation
        self.results[op] = "pass"
        self.outputs[op] = ""
        # Temporary storage
        self.tmp = {
            "printed_header": False,  # Header for this op has not been printed yet
            "op": op,  # Currently investigated operation
            "res": verbosity_levels["pass"],  # Current result status
            "name": op.name if isinstance(op, qml.operation.Operation) else op.__name__,
        }
        try:
            self.check_single_operation(op, parameters, wires)
        except CheckerError:
            pass

        # Store the result for this operation outside of tmp and print summary
        self.results[op] = levels_verbosity[self.tmp["res"]]
        if self.results[op] == "pass":
            self.print_(
                f"No problems have been found with the operation {self.tmp['name']}.\n", "pass"
            )

        return self.results[op], self.outputs[op]

    def print_(self, string, level=None):
        """Print a string if the verbosity level allows it, color it if applicable,
        and increment the result status for the currently checked operation if necessary.

        Args:
            string (str): String to be printed
            level (str): One of the verbosity levels (see class documentation)
                If the level is in the levels that are printed, print the string to console
                and store it in ``self.outputs``.

        Returns:

        A header is printed whenever a ``print_`` statement is executed first for
        a given operation (and the verbosity levels actually allow for an output).
        """

        self.tmp["res"] = min(self.tmp["res"], verbosity_levels[level])

        # Colorize the string
        if self.print_color and level is not None:
            string = f"\033[{_colors[level]}m{string}\033[0m"

        # Errors are always printed
        if level == "error" or level in self._verbosity:
            if not self.tmp["printed_header"]:
                header = f"Checking operation {self.tmp['name']} for consistency.\n" + "= " * 40
                self.print_fn(header)
                self.outputs[self.tmp["op"]] += header
                self.tmp["printed_header"] = True
            self.print_fn(string)
            self.outputs[self.tmp["op"]] += "\n" + string

    def check_single_operation(self, op, parameters, wires):
        """Check one operation subclass to define all required properties,
        be well-defined, and have consistent properties.

        Args:
            op (type): Operation to check, may be a class or and instance.
            parameters (Sequence[int or float]): Parameters with which the operation(s)
                is/are expected to work.
            wires (.wires.Wires): Wires with which the operation(s) is/are expected to work.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        # Retrieve parameters and wires if op is operation instance instead of class
        if not inspect.isclass(op):
            parameters = op.parameters
            wires = op.wires
            op = type(op)

        wires = self._check_wires(op, wires)
        parameters = self._check_parameters(op, parameters)

        # Check class instantiation
        self._check_instantiation(op, parameters, wires)

        # Check methods to work with the same number of parameters as instantiation
        for method_tuple in _default_methods_to_check:
            self._check_single_method(op, method_tuple, parameters, wires)
        parameters = [par for par in parameters if len(par) in self.tmp["possible_num_params"]]

        self._check_properties(op, parameters, wires)
        self._check_decompositions(op, parameters, wires)
        if issubclass(op, qml.operation.Operation):
            self._check_parameter_shift(op, parameters, wires)

        self._check_derivatives(op, parameters, wires)

    def _check_wires(self, op, wires):
        """Check that ``num_wires`` is defined, that provided wires match the given number
        and otherwise create correct number of wires.
        TODO: Check whether the following is reasonable:
        If ``num_wires`` is ``AnyWires``, its size is undetermined and we default to 2
        wires.
        """
        if isinstance(op.num_wires, property):
            self.print_(
                f"The operation {self.tmp['name']} does not define the number of wires it acts on.",
                "fatal_error",
            )
            raise CheckerError("Fatal error: Subsequent checks will not be possible.")

        if wires is None:
            if op.num_wires != AnyWires:
                wires = qml.wires.Wires(range(op.num_wires))
            else:
                # Use a dummy case of 2 wires for operations with flexible number of wires
                wires = qml.wires.Wires([0, 1])
        else:
            if op.num_wires != AnyWires and len(wires) != op.num_wires:
                self.print_(
                    f"The number of provided wires ({len(wires)}) does not match the expected "
                    f"number ({op.num_wires}) for operation {self.tmp['name']}",
                    "fatal_error",
                )
                raise CheckerError("Fatal error: Subsequent checks will not be possible.")

        return wires

    def _check_parameters(self, op, parameters):
        """Check whether ``num_params`` is defined, that provided parameters
        match that number if it is defined, and otherwise create parameter
        sets of length ``0`` to ``self.max_num_params``.
        """
        num_params_known = isinstance(op.num_params, int)
        self.tmp["num_params_known"] = num_params_known
        if parameters is None:
            if num_params_known:
                parameters = [np.random.random(op.num_params)]
            else:
                parameters = [np.random.random(num) for num in range(self.max_num_params + 1)]
        elif num_params_known and len(parameters) != op.num_params:
            self.print_(
                f"The number of provided parameters ({len(parameters)}) does not match "
                f"the expected number ({op.num_params}) for operation {self.tmp['name']}",
                "fatal_error",
            )
            raise CheckerError("Fatal error: Subsequent checks will not be possible.")
        else:
            parameters = [parameters]

        return parameters

    def _check_instantiation(self, op, parameters, wires):
        """Check whether instantiation of an operation works, either
        with provided parameters and wires, or with a series of numbers
        of parameters. The number(s) of parameters with which instantiation
        works is stored in ``self.tmp["possible_num_params"]``."""
        if self.tmp["num_params_known"]:
            op(*parameters[0], wires=wires)
            self.tmp["possible_num_params"] = [op.num_params]
            return

        possible_num_params = []
        for par in parameters:
            try:
                op(*par, wires=wires)
                possible_num_params.append(len(par))
            except TypeError:
                pass

        if len(possible_num_params) == 1 and len(parameters) > 1:
            self.print_(
                f"Instantiating {self.tmp['name']} only succeeded when using "
                f"{possible_num_params[0]} parameter(s).\n"
                "Consider specifying the number of parameters by setting op.num_params.",
                "hint",
            )
        elif not possible_num_params:
            par_lens = [len(par) for par in parameters]
            err_str = (
                f"Instantiating {self.tmp['name']} did not succeed with any of\n"
                f"{par_lens} parameters."
            )
            if len(parameters) == 1:
                err_str += (
                    "\nIt seems that you provided parameters of the wrong length "
                    "for this operation,\ncheck the input to check_operation."
                )
            self.print_(err_str, "fatal_error")

        self.tmp["possible_num_params"] = possible_num_params

    def _check_single_method(self, op, method_tuple, parameters, wires):
        """Check whether a specific method of an operation works with
        provided parameters and wires, or with the same number of
        parameters as the instantiation allowed."""

        method, expected_exc, use_wires = method_tuple
        wrapped_method = wrap_op_method(op, method, expected_exc)
        kwargs = {"wires": wires} if use_wires else {}
        if self.tmp["num_params_known"]:
            par = parameters[0]
            exc = wrapped_method(*par, **kwargs)
            if not isinstance(exc, Exception):
                # If no or the expected exception occured, return
                return

            # It might be that the "compute_..." method requires
            # different args than __init__ but that this is accomodated
            # for in the hyperparameters of the operation.
            try:
                instance = op(*par, wires=wires)
                getattr(instance, method.replace("compute", "get"))()
                self.print_(exc, "comment")
                self.print_(
                    f"Operation method {self.tmp['name']}.{method} does not work\n"
                    f"with num_params ({op.num_params}) parameters (see above) but is "
                    "using additional (hyper)parameters.",
                    "comment",
                )
                # If the above indeed is the case, return
                return
            except Exception as f:  # pylint: disable=broad-except
                self.print_(exc, "error")
                self.print_(f, "error")

            self.print_(
                f"Operation method {self.tmp['name']}.{method} does not work\n"
                f"with num_params ({op.num_params}) parameters.",
                "fatal_error" if method == "compute_matrix" else "error",
            )
            if method == "compute_matrix":
                raise CheckerError("Fatal error: Subsequent checks will not be possible.")
            return

        failing_nums = []
        succeeding_nums = []
        for par in parameters:
            exc = wrapped_method(*par, **kwargs)
            num = len(par)
            if (
                not (exc is None or isinstance(exc, Exception))
                and num not in self.tmp["possible_num_params"]
            ):
                succeeding_nums.append(num)
            elif isinstance(exc, Exception) and num in self.tmp["possible_num_params"]:
                failing_nums.append(num)

        if failing_nums:
            self.print_(
                f"Operation method {self.tmp['name']}.{method} does not work\n"
                f"with number(s) of parameters {failing_nums}\n"
                "but instantiation works with this/these number(s) of parameters.",
                "fatal_error" if method == "compute_matrix" else "error",
            )
            if method == "compute_matrix":
                raise CheckerError("Fatal error: Subsequent checks will not be possible.")

        if succeeding_nums:
            self.print_(
                f"Operation method {self.tmp['name']}.{method} works\n"
                f"with number(s) of parameters {succeeding_nums}\n"
                "but instantiation does not work with this/these number(s) of parameters.",
                "comment",
            )

    def _check_decompositions(self, op, parameters, wires):
        """Check that all defined decompositions work and yield the same matrix."""
        for par in parameters:
            matrices = [meth(op, par, wires) for meth in decomposition_methods]
            matrices = [mat for mat in matrices if mat is not None]
            for mat in matrices[1:]:
                # TODO: improvement: Take a guess at _which_ of the matrices is/ are wrong
                if not equal_up_to_phase(matrices[0], mat, atol=self.tol):
                    self.print_(
                        f"Matrices do not coincide for {self.tmp['name']}.",
                        # f"\n{np.round(matrices[0], 5)}\n{np.round(mat, 5)}",
                        "error",
                    )

    def _check_properties(self, op, parameters, wires):
        """Check basic properties that need to be satisfied as well as the correctness
        of additional properties that are given by attributes of the operation."""
        for par in parameters:
            instance = op(*par, wires=wires)
            # Check that the matrix is square and has the correct size for op.num_wires
            mat = wrap_op_method(instance, "get_matrix", MatrixUndefinedError)()
            self._check_matrix_shape(mat, op)
            # Check that the eigenvalues are produced correctly
            eigvals = wrap_op_method(instance, "get_eigvals", (EigvalsUndefinedError, TypeError))()
            self._check_eigvals(eigvals, mat)
            # Check that the diagonalizing gates diagonalize the operation matrix
            diag_gates = wrap_op_method(
                instance, "diagonalizing_gates", (DiagGatesUndefinedError, TypeError)
            )()
            self._check_diag_gates(diag_gates, mat, eigvals)
            # Check that the basis is given correctly
            self._check_basis(mat, instance)

    def _check_matrix_shape(self, matrix, op):
        """Check that a matrix attributed to an operation has the correct shape."""
        if matrix is None:
            return
        if not matrix.shape[0] == matrix.shape[1]:
            self.print_(
                f"The operation {self.tmp['name']} defines a non-square matrix.", "fatal_error"
            )
            raise CheckerError("Fatal error: Subsequent checks will not be possible.")
        mat_num_wires = int(np.log2(matrix.shape[0]))
        if not mat_num_wires == op.num_wires and op.num_wires != AnyWires:
            self.print_(
                f"The operation {self.tmp['name']} defines a matrix for {mat_num_wires} wires but "
                f"is defined to have {op.num_wires} wires.",
                "fatal_error",
            )
            raise CheckerError("Fatal error: Subsequent checks will not be possible.")
        return

    def _check_eigvals(self, eigvals, matrix):
        """Check that produced eigvals for an operation coincide with the
        eigvals of a matrix representation of the same operation."""
        if matrix is None or eigvals is None:
            return
        mat_eigvals = np.sort(np.round(np.linalg.eigvals(matrix), 14))
        if not np.allclose(mat_eigvals, np.sort(np.round(eigvals, 14))):
            self.print_(
                f"The eigenvalues of the matrix and the stored eigvals for {self.tmp['name']} "
                "do not match.",
                "error",
            )
        return

    def _check_diag_gates(self, diag_gates, matrix, eigvals):
        """Check that the diagonalizing gates attributed to an operation
        produce a diagonal matrix, and that it has the correct eigenvalues."""
        if diag_gates is None or matrix is None:
            return
        if diag_gates == []:
            diag_mat = np.eye(matrix.shape[0])
        else:
            with qml.tape.QuantumTape() as tape:
                [op.queue() for op in diag_gates]  # pylint: disable=expression-not-assigned
            diag_mat = qml.matrix(tape)

        diagonalized = diag_mat @ matrix @ diag_mat.conj().T
        if not is_diagonal(diagonalized, atol=self.tol):
            self.print_(
                f"The diagonalizing gates do not diagonalize the matrix for {self.tmp['name']}.",
                "error",
            )
            return
        if eigvals is not None and not np.allclose(
            np.sort(np.round(eigvals, 14)), np.sort(np.round(np.diag(diagonalized), 14))
        ):
            self.print_(
                "The diagonalizing gates diagonalize the matrix but produce wrong "
                f"eigenvalues for {self.tmp['name']}.",
                "error",
            )
        return

    def _check_basis(self, matrix, instance):
        """Check that a matrix attributed to an operation is diagonal in the basis
        indicated by that operation's ``basis`` property."""
        try:
            basis = instance.basis
        except AttributeError:
            basis = None

        if basis is None or matrix is None:
            return

        if basis == "X":
            diag_gates = [qml.Hadamard]
        elif basis == "Y":
            diag_gates = [qml.PauliZ, qml.S, qml.Hadamard]
        elif basis == "Z":
            diag_gates = [qml.Identity]

        target_wires = qml.wires.Wires.unique_wires([instance.wires, instance.control_wires])
        with qml.tape.QuantumTape() as tape:
            for w in target_wires:
                # pylint: disable=expression-not-assigned
                [diag_gate(wires=w) for diag_gate in diag_gates]

        diag_mat = qml.operation.expand_matrix(qml.matrix(tape), target_wires, instance.wires)
        if not is_diagonal(diag_mat @ matrix @ diag_mat.conj().T, atol=self.tol):
            self.print_(
                f"The operation {self.tmp['name']} is not diagonal in the provided basis",
                "error",
            )

    def _check_derivatives(self, op, parameters, wires):
        """Check that the matrix representation of an operation obtained via qml.matrix
        is differentiable in the autograd interface."""
        for par in parameters:
            num = len(par)
            if num not in self.tmp["possible_num_params"] or num == 0:
                continue
            try:
                qml.matrix(op(*par, wires=wires))
            except qml.operation.MatrixUndefinedError:
                continue
            if getattr(op, "grad_method", None) is None:
                continue
            instance_fn = lambda *args: qml.matrix(op(*args, wires=wires))

            par_autograd = tuple(
                p if isinstance(p, str) else np.array(p, requires_grad=True) for p in par
            )
            autograd_jac = complex_jacobian(instance_fn)(*par_autograd)
            if num == 1:
                autograd_jac = (autograd_jac,)
            # TODO: allow for string arguments by setting argnum
            if has_jax and not any(isinstance(p, str) for p in par):
                par_jax = tuple(jax.numpy.array(p) for p in par)
                jax_jacobian = partial(jax.jacobian, argnums=list(range(num)))
                jax_jac = complex_jacobian(instance_fn, jax_jacobian)(*par_jax)
                if num == 1:
                    jax_jac = (jax_jac,)
                if not all(qml.math.allclose(ag, j) for ag, j in zip(autograd_jac, jax_jac)):
                    self.print_(
                        f"The jacobian of the matrix for {self.tmp['name']} does not match between\n"
                        "the autograd and jax interfaces."
                        f"\n{autograd_jac}\n{jax_jac}",
                        "error",
                    )
            # TODO: allow for string arguments by setting argnum
            if has_torch and not any(isinstance(p, str) for p in par):
                par_torch = tuple(torch.tensor(p, requires_grad=True) for p in par)
                torch_jac = complex_jacobian(instance_fn, torch_jacobian)(*par_torch)
                if not all(qml.math.allclose(ag, t) for ag, t in zip(autograd_jac, torch_jac)):
                    self.print_(
                        f"The jacobian of the matrix for {self.tmp['name']} does not match between\n"
                        "the autograd and torch interfaces.",
                        "error",
                    )
            # TODO: allow for string arguments by setting argnum
            if has_tf and not any(isinstance(p, str) for p in par):
                par_tf = tuple(tf.Variable(p) for p in par)
                tf_jac = complex_jacobian(instance_fn, tf_jacobian)(*par_tf)
                if not all(qml.math.allclose(ag, t) for ag, t in zip(autograd_jac, tf_jac)):
                    self.print_(
                        f"The jacobian of the matrix for {self.tmp['name']} does not match between\n"
                        "the autograd and tensorflow interfaces.",
                        "error",
                    )

    def _check_parameter_shift(self, op, parameters, wires):
        """Check that an operation is differentiable if it is marked to be, and that
        the correct derivative is produced. Note that this can be computationally expensive."""
        if issubclass(op, qml.operation.Channel):
            self.print_("Channels cannot be checked for the correct derivative yet.", "hint")
            return

        if self.tmp["num_params_known"] and op.num_params != 1:
            # Operation has more than one parameter. Skip it
            return

        grad_method = getattr(op, "grad_method", None)
        if grad_method is None:
            return
        grad_method_is_hardcoded = isinstance(grad_method, str)
        has_grad_recipe = (
            getattr(op, "grad_recipe", None) is not None
            and op.grad_recipe != [None] * self.tmp["possible_num_params"][0]
        )
        if grad_method_is_hardcoded and op.grad_method != "A" and has_grad_recipe:
            self.print_(
                f"A grad_recipe is provided for {self.tmp['name']} but grad_method is "
                f"{op.grad_method}. Consider changing it to 'A'.",
                "hint",
            )
        elif op.grad_method == "F":
            return

        for par in parameters:
            instance = op(*par, wires=wires)
            try:
                freq = instance.parameter_frequencies[0]
                coeffs, shifts = qml.gradients.generate_shift_rule(freq)
                std_recipe = np.array([coeffs, np.ones_like(coeffs), shifts])
            except qml.operation.ParameterFrequenciesUndefinedError:
                std_recipe = None
            recipe = np.array(op.grad_recipe[0]).T if has_grad_recipe else std_recipe

            if recipe is None:
                continue
            try:
                correct_recipe = recipe_yields_commutator(recipe, op, par, wires)
            except qml.operation.MatrixUndefinedError:
                continue

            if not correct_recipe:
                self.print_(
                    f"The grad_recipe of {self.tmp['name']} does not yield the correct "
                    "derivative.",
                    "error",
                )
            if has_grad_recipe and np.allclose(recipe, std_recipe):
                self.print_(
                    f"The grad_recipe of {self.tmp['name']} is a standard parameter-"
                    "shift rule. Consider removing it and adding parameter_frequencies "
                    f"to the operation instead in order to allow for flexible shift values.",
                    "hint",
                )
