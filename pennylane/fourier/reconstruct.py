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
"""Contains a function that computes the fourier series equivalent to
a quantum expectation value."""
#from itertools import chain, combinations
from functools import wraps
import numpy as np
import pennylane as qml

def _reconstruct_gen(fun, spectrum, shifts, fun_at_zero=None):
    # Take care of shifts close to zero if fun_at_zero was provided
    if fun_at_zero is not None:
        close_to_zero = np.isclose(shifts, 0.0)
        if any(close_to_zero):
            zero_in_shifts = True
            zero_idx = np.where(close_to_zero)
            shifts = np.concatenate([shifts[zero_idx], shifts[:zero_idx], shifts[zero_idx+1:]])
        else:
            zero_in_shifts = False
        
    L = len(shifts)
    # Construct the coefficient matrix case by case
    C1 = np.ones((L, 1))
    C2 = np.cos(np.outer(shifts, spectrum))
    C3 = np.sin(np.outer(shifts, spectrum))
    C = np.hstack([C1, C2, C3])

    # Evaluate the function to reconstruct at the shifted positions
    if fun_at_zero is not None and zero_in_shifts:
        evals = np.array([fun_at_zero]+list(map(fun, shifts[1:])))
    else:
        evals = np.array(list(map(fun, shifts)))

    # Solve the system of linear equations
    W = np.linalg.solve(C, evals)

    # Extract the Fourier coefficients
    R = (L-1)//2
    a0 = W[0]
    a = W[1 : R + 1]
    b = W[R + 1 :]

    # Construct the Fourier series
    def _reconstruction(x):
        """Univariate reconstruction based on arbitrary shifts."""
        return a0 + np.dot(a, np.cos(spectrum * x)) + np.dot(b, np.sin(spectrum * x))

    return _reconstruction


def _reconstruct_equ(fun, num_frequency, fun_at_zero=None):
    r"""Reconstruct a univariate trigonometric function with consecutive integer 
    frequencies, using trigonometric interpolation and equidistant shifts.

    This technique is based on 
    `Dirichlet kernels <https://en.wikipedia.org/wiki/Dirichlet_kernel>`_, see 
    `Vidal and Theis (2018) <https://arxiv.org/abs/1812.06323>`_ or
    `Wierichs et al. (2021) <https://arxiv.org/abs/2107.12390>`_.

    Args:
        fun (callable): the function to reconstruct
        num_frequency (int): the number of integer frequencies present in ``fun``.
        fun_at_zero (float): The value of ``fun`` at 0. Computed if not provided.

    Returns:
        callable: The reconstruction function with ``num_frequency`` frequencies,
        coinciding with ``fun`` on the same number of points.
    """
    a, b = (num_frequency + 0.5) / np.pi, 0.5 / np.pi
    shifts_pos = np.arange(1, num_frequency + 1) / a
    shifts_neg = -shifts_pos[::-1]
    fun_at_zero = float(fun(0.0)) if fun_at_zero is None else fun_at_zero
    evals = list(map(fun, shifts_neg)) + [fun_at_zero] + list(map(fun, shifts_pos))
    shifts = np.concatenate([shifts_neg, [0.0], shifts_pos])
    def _reconstruction(x):
        """Univariate reconstruction based on equidistant shifts and Dirichlet kernels."""
        return np.dot(evals, np.sinc(a * (x-shifts)) / np.sinc(b*(x-shifts)))

    return _reconstruction

def reconstruct(qnode, _spectra=None, order=1, shifts=None, nums_frequency=None, ids="auto", decimals=5):

    @wraps(qnode)
    def wrapper(*args, **kwargs):
        # what about ids in the following?
        #print(args)
        #print(kwargs)
        class_jac = qml.transforms.classical_jacobian(qnode, ids)(*args, **kwargs)
        if nums_frequency is None:
            if _spectra is None:
                _spectra = qml.fourier.spectrum(qnode, encoding_gates=ids, decimals=decimals)(*args, **kwargs)
            #else:
                #qnode.construct(args, kwargs)
            if shifts is None:
                shifts = {}
            uses_fun_at_zero = {id: any(np.isclose(_shifts, 0.0)) for id, _shifts in shifts.items()}
            for id in set(_spectra.keys())-set(shifts.keys()):
                R = len(_spectra[id])
                shifts[id] = np.arange(-R, R+1)*2*np.pi/(2*R+1)
                uses_fun_at_zero[id] = True
            if any(uses_fun_at_zero.values()):
                fun_at_zero = qnode(*args, **kwargs)
            _recon_fn = _reconstruct_gen
            _recon_kwargs = {
                id: {
                    'spectrum': spectrum,
                    'shifts': shifts[id],
                    'fun_at_zero': fun_at_zero if uses_fun_at_zero[id] else None,
                }
                for id, spectrum in _spectra.items()
            }

        else:
            fun_at_zero = qnode(*args, **kwargs)
            _recon_fn = _reconstruct_equ
            _recon_kwargs = {
                id: {
                    'num_frequency': num_frequency,
                    'fun_at_zero': fun_at_zero,
                }
                for id, num_frequency in nums_frequency.items()
            }

        tape = qnode.qtape
        par_idxs = {}
        for j, jac_row in zip(tape.trainable_params, class_jac):
            op = tape._par_info[j]["op"]
            id = op.id
            jac_idx = np.where(jac_row)[0]
            if len(jac_idx)==0:
                if id not in par_idxs:
                    par_idxs[id] = None
            elif len(jac_idx)==1:
                if id in par_idxs:
                    if par_idxs[id]!=jac_idx[0]:
                        raise ValueError(
                            "Gates with the same id are expected to be controlled by the "
                            f"same parameter. Id {id} is (at least) controlled by parameters "
                            f"{par_idxs[id]} and {jac_idx[0]}."
                        )
                else:
                    par_idxs[id] = jac_idx[0]
            else:
                raise ValueError(
                    f"Multiple parameters feed into the operation {op.name} in "
                    "the provided qnode; only operations controlled by a single parameter are "
                    "supported."
                )

        if len(set(par_idxs.values()))!=len(par_idxs.values()):
            raise ValueError("Each parameter is expected to control gates with one id")

        def constant_fn(x):
            return fun_at_zero

        reconstructions = {}
        for id in par_idxs:
            par_idx = par_idxs[id]
            if par_idx is None:
                reconstructions[id] = constant_fn
                continue
            for arg_idx, arg in enumerate(args):
                l = len(qml._flatten(arg))
                if l>par_idx:
                    break
                par_idx -= l
            vec = np.eye(l)[par_idx]
            def _univariate_fn(x):
                new_arg = qml._unflatten(qml._flatten(args[arg_idx])+vec*x, args[arg_idx])
                new_args = ((arg if j!=arg_idx else new_arg) for j, arg in enumerate(args))
                return qnode(*new_args, **kwargs)
            
            reconstructions[id] = _recon_fn(_univariate_fn, **_recon_kwargs)

        return reconstructions

    return wrapper

        
