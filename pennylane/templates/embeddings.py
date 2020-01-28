# Copyright 2018-2019 Xanadu Quantum Technologies Inc.

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
Embeddings are templates that take features and encode them into a quantum state.
They can optionally be repeated, and may contain trainable parameters. Embeddings are typically
used at the beginning of a circuit.
"""
#pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import numpy as np
from pennylane.templates.decorator import template
from pennylane.ops import RX, RY, RZ, CNOT, Hadamard, BasisState, Squeezing, Displacement, QubitStateVector
from pennylane.templates.utils import (_check_shape, _check_no_variable, _check_wires,
                                       _check_is_in_options, _check_type,
                                       _check_number_of_layers, _get_shape)
from pennylane.variable import VariableRef


TOLERANCE = 1e-3

@template
def AmplitudeEmbedding(features, wires, pad=None, normalize=False):
    r"""Encodes :math:`2^n` features into the amplitude vector of :math:`n` qubits.

    By setting ``pad`` to a real or complex number, ``features`` is automatically padded to dimension
    :math:`2^n` where :math:`n` is the number of qubits used in the embedding.

    To represent a valid quantum state vector, the L2-norm of ``features`` must be one.
    The argument ``normalize`` can be set to ``True`` to automatically normalize the features.

    If both automatic padding and normalization are used, padding is executed *before* normalizing.

    .. note::

        On some devices, ``AmplitudeEmbedding`` must be the first operation of a quantum node.


    .. note::

        ``AmplitudeEmbedding`` calls a circuit that involves non-trivial classical processing of the
        features. The ``features`` argument is therefore **not differentiable** when using the template, and
        gradients with respect to the features cannot be computed by PennyLane.

    .. warning::

        ``AmplitudeEmbedding`` calls a circuit that involves non-trivial classical processing of the
        features. The `features` argument is therefore not differentiable when using the template, and
        gradients with respect to the argument cannot be computed by PennyLane.

    Args:
        features (array): input array of shape ``(2^n,)``
        wires (Sequence[int] or int): :math:`n` qubit indices that the template acts on
        pad (float or complex): if not None, the input is padded with this constant to size :math:`2^n`
        normalize (Boolean): controls the activation of automatic normalization

    Raises:
        ValueError: if inputs do not have the correct format

    .. UsageDetails::

        Amplitude embedding encodes a normalized :math:`2^n`-dimensional feature vector into the state
        of :math:`n` qubits:

        .. code-block:: python

            import pennylane as qml
            from pennylane.templates import AmplitudeEmbedding

            dev = qml.device('default.qubit', wires=2)

            @qml.qnode(dev)
            def circuit(f=None):
                AmplitudeEmbedding(features=f, wires=range(2))
                return qml.expval(qml.PauliZ(0))

            circuit(f=[1/2, 1/2, 1/2, 1/2])

        Checking the final state of the device, we find that it is equivalent to the input passed to the circuit:

        >>> dev._state
        [0.5+0.j 0.5+0.j 0.5+0.j 0.5+0.j]

        **Passing features as positional arguments to a quantum node**

        The ``features`` argument of ``AmplitudeEmbedding`` can in principle also be passed to the quantum node
        as a positional argument:

        .. code-block:: python

            @qml.qnode(dev)
            def circuit(f):
                AmplitudeEmbedding(features=f, wires=range(2))
                return qml.expval(qml.PauliZ(0))

        However, due to non-trivial classical processing to construct the state preparation circuit,
        the features argument is **not differentiable**.

        >>> g = qml.grad(circuit, argnum=0)
        >>> g([1,1,1,1])
        ValueError: Cannot differentiate wrt parameter(s) {0, 1, 2, 3}.


        **Normalization**

        The template will raise an error if the feature input is not normalized.
        One can set ``normalize=True`` to automatically normalize it:

        .. code-block:: python

            @qml.qnode(dev)
            def circuit(f=None):
                AmplitudeEmbedding(features=f, wires=range(2), normalize=True)
                return qml.expval(qml.PauliZ(0))

            circuit(f=[15, 15, 15, 15])

        The re-normalized feature vector is encoded into the quantum state vector:

        >>> dev._state
        [0.5 + 0.j, 0.5 + 0.j, 0.5 + 0.j, 0.5 + 0.j]

        **Padding**

        If the dimension of the feature vector is smaller than the number of amplitudes,
        one can automatically pad it with a constant for the missing dimensions using the ``pad`` option:

        .. code-block:: python

            from math import sqrt

            @qml.qnode(dev)
            def circuit(f=None):
                AmplitudeEmbedding(features=f, wires=range(2), pad=0.)
                return qml.expval(qml.PauliZ(0))

            circuit(f=[1/sqrt(2), 1/sqrt(2)])

        >>> dev._state
        [0.70710678 + 0.j, 0.70710678 + 0.j, 0.0 + 0.j, 0.0 + 0.j]

        **Operations before the embedding**

        On some devices, ``AmplitudeEmbedding`` must be the first operation in the quantum node.
        For example, ``'default.qubit'`` complains when running the following circuit:

        .. code-block:: python

            dev = qml.device('default.qubit', wires=2)

            @qml.qnode(dev)
            def circuit(f=None):
                qml.Hadamard(wires=0)
                AmplitudeEmbedding(features=f, wires=range(2))
                return qml.expval(qml.PauliZ(0))


        >>> circuit(f=[1/2, 1/2, 1/2, 1/2])
        pennylane._device.DeviceError: Operation QubitStateVector cannot be used
        after other Operations have already been applied on a default.qubit device.

    """

    #############
    # Input checks

    _check_no_variable(pad, msg="'pad' cannot be differentiable")
    _check_no_variable(normalize, msg="'normalize' cannot be differentiable")

    wires = _check_wires(wires)

    n_amplitudes = 2**len(wires)
    expected_shape = (n_amplitudes,)
    if pad is None:
        shape = _check_shape(features, expected_shape, msg="'features' must be of shape {}; got {}. Use the 'pad' "
                                                           "argument for automated padding."
                                                           "".format(expected_shape, _get_shape(features)))
    else:
        shape = _check_shape(features, expected_shape, bound='max', msg="'features' must be of shape {} or smaller "
                                                                      "to be padded; got {}"
                                                                      "".format(expected_shape, _get_shape(features)))

    _check_type(pad, [float, complex, type(None)], msg="'pad' must be a float or complex; got {}".format(pad))
    _check_type(normalize, [bool], msg="'normalize' must be a boolean; got {}".format(normalize))

    ###############

    #############
    # Preprocessing

    # pad
    n_features = shape[0]
    if pad is not None and n_amplitudes > n_features:
        features = np.pad(features, (0, n_amplitudes-n_features), mode='constant', constant_values=pad)

    # normalize
    if isinstance(features[0], VariableRef):
        feature_values = [s.val for s in features]
        norm = np.sum(np.abs(feature_values)**2)
    else:
        norm = np.sum(np.abs(features)**2)

    if not np.isclose(norm, 1.0, atol=TOLERANCE, rtol=0):
        if normalize or pad:
            features = features/np.sqrt(norm)
        else:
            raise ValueError("'features' must be a vector of length 1.0; got length {}."
                             "Use 'normalization=True' to automatically normalize.".format(norm))

    ###############

    features = np.array(features)
    QubitStateVector(features, wires=wires)


@template
def AngleEmbedding(features, wires, rotation='X'):
    r"""
    Encodes :math:`N` features into the rotation angles of :math:`n` qubits, where :math:`N \leq n`.

    The rotations can be chosen as either :class:`~pennylane.ops.RX`, :class:`~pennylane.ops.RY`
    or :class:`~pennylane.ops.RZ` gates, as defined by the ``rotation`` parameter:

    * ``rotation='X'`` uses the features as angles of RX rotations

    * ``rotation='Y'`` uses the features as angles of RY rotations

    * ``rotation='Z'`` uses the features as angles of RZ rotations

    The length of ``features`` has to be smaller or equal to the number of qubits. If there are fewer entries in
    ``features`` than rotations, the circuit does not apply the remaining rotation gates.

    Args:
        features (array): input array of shape ``(N,)``, where N is the number of input features to embed,
            with :math:`N\leq n`
        wires (Sequence[int] or int): qubit indices that the template acts on
        rotation (str): Type of rotations used

    Raises:
        ValueError: if inputs do not have the correct format
    """

    #############
    # Input checks

    _check_no_variable(rotation, msg="'rotation' cannot be differentiable")

    wires = _check_wires(wires)

    _check_shape(features, (len(wires),), bound='max', msg="'features' must be of shape {} or smaller; "
                                                           "got {}.".format((len(wires),), _get_shape(features)))
    _check_type(rotation, [str], msg="'rotation' must be a string; got {}".format(rotation))

    _check_is_in_options(rotation, ['X', 'Y', 'Z'], msg="did not recognize option {} for 'rotation'.".format(rotation))

    ###############

    if rotation == 'X':
        for f, w in zip(features, wires):
            RX(f, wires=w)
    elif rotation == 'Y':
        for f, w in zip(features, wires):
            RY(f, wires=w)
    elif rotation == 'Z':
        for f, w in zip(features, wires):
            RZ(f, wires=w)


@template
def BasisEmbedding(features, wires):
    r"""Encodes :math:`n` binary features into a basis state of :math:`n` qubits.

    For example, for ``features=np.array([0, 1, 0])``, the quantum system will be
    prepared in state :math:`|010 \rangle`.

    .. warning::

        ``BasisEmbedding`` calls a circuit whose architecture depends on the binary features.
        The ``features`` argument is therefore not differentiable when using the template, and
        gradients with respect to the argument cannot be computed by PennyLane.

    Args:
        features (array): binary input array of shape ``(n, )``
        wires (Sequence[int] or int): qubit indices that the template acts on

    Raises:
        ValueError: if inputs do not have the correct format
    """

    #############
    # Input checks

    wires = _check_wires(wires)

    expected_shape = (len(wires),)
    _check_shape(features, expected_shape, msg="'features' must be of shape {}; got {}"
                                               "".format(expected_shape, _get_shape(features)))

    if any([b not in [0, 1] for b in features]):
        raise ValueError("'basis_state' must only consist of 0s and 1s; got {}".format(features))

    ###############

    features = np.array(features)
    BasisState(features, wires=wires)


def _qaoa_feature_encoding_hamiltonian(features, n_features, wires):
    """Implements the encoding Hamiltonian of the QAOA embedding.

    Args:
        features (array): array of features to encode
        n_features (int): number of features to encode
    """
    for idx, w in enumerate(wires):
        # Either feed in feature
        if idx < n_features:
            RX(features[idx], wires=w)
        # or a Hadamard
        else:
            Hadamard(wires=w)


def _qaoa_ising_hamiltonian(weights, wires, local_fields, l):
    """Implements the Ising-like Hamiltonian of the QAOA embedding.

    Args:
        weights (array): array of weights
        wires (Sequence[int] or int): `n` qubit indices that the template acts on
        local_fields (str): gate implementing the local field
        l (int): layer index
    """
    # trainable "Ising" ansatz
    if len(wires) == 1:
        local_fields(weights[l][0], wires=wires[0])

    elif len(wires) == 2:
        # ZZ coupling
        CNOT(wires=[wires[0], wires[1]])
        RZ(2 * weights[l][0], wires=wires[0])
        CNOT(wires=[wires[0], wires[1]])

        # local fields
        for i, _ in enumerate(wires):
            local_fields(weights[l][i + 1], wires=wires[i])

    else:
        for i, _ in enumerate(wires):
            if i < len(wires) - 1:
                # ZZ coupling
                CNOT(wires=[wires[i], wires[i + 1]])
                RZ(2 * weights[l][i], wires=wires[i])
                CNOT(wires=[wires[i], wires[i + 1]])
            else:
                # ZZ coupling to enforce periodic boundary condition
                CNOT(wires=[wires[i], wires[0]])
                RZ(2 * weights[l][i], wires=wires[i])
                CNOT(wires=[wires[i], wires[0]])
        # local fields
        for i, _ in enumerate(wires):
            local_fields(weights[l][len(wires) + i], wires=wires[i])


@template
def QAOAEmbedding(features, weights, wires, local_field='Y'):
    r"""
    Encodes :math:`N` features into :math:`n>N` qubits, using a layered, trainable quantum
    circuit that is inspired by the QAOA ansatz.

    A single layer applies two circuits or "Hamiltonians": The first encodes the features, and the second is
    a variational ansatz inspired by a 1-dimensional Ising model. The feature-encoding circuit associates features with
    the angles of :class:`RX` rotations. The Ising ansatz consists of trainable two-qubit ZZ interactions
    :math:`e^{-i \alpha \sigma_z \otimes \sigma_z}`,
    and trainable local fields :math:`e^{-i \frac{\beta}{2} \sigma_{\mu}}`, where :math:`\sigma_{\mu}`
    can be chosen to be :math:`\sigma_{x}`, :math:`\sigma_{y}` or :math:`\sigma_{z}`
    (default choice is :math:`\sigma_{y}` or the ``RY`` gate), and :math:`\alpha, \beta` are adjustable gate parameters.

    The number of features has to be smaller or equal to the number of qubits. If there are fewer features than
    qubits, the feature-encoding rotation is replaced by a Hadamard gate.

    The argument ``weights`` contains an array of the :math:`\alpha, \beta` parameters for each layer.
    The number of layers :math:`L` is derived from the first dimension of ``weights``, which has the following
    shape:

    * :math:`(L, )`, if the embedding acts on a single wire,
    * :math:`(L, 3)`, if the embedding acts on two wires,
    * :math:`(L, 2n)` else.

    After the :math:`L` th layer, another set of feature-encoding :class:`RX` gates is applied.

    This is an example for the full embedding circuit using 2 layers, 3 features, 4 wires, and ``RY`` local fields:

    |

    .. figure:: ../../_static/qaoa_layers.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    |

    .. note::
        ``QAOAEmbedding`` supports gradient computations with respect to both the ``features`` and the ``weights``
        arguments. Note that trainable parameters need to be passed to the quantum node as positional arguments.

    Args:
        features (array): array of features to encode
        weights (array): array of weights
        wires (Sequence[int] or int): `n` qubit indices that the template acts on
        local_field (str): type of local field used, one of ``'X'``, ``'Y'``, or ``'Z'``

    Raises:
        ValueError: if inputs do not have the correct format

    .. UsageDetails::

        The QAOA embedding encodes an :math:`n`-dimensional feature vector into at most :math:`n` qubits. The
        embedding applies layers of a circuit, and each layer is defined by a set of weight parameters.

        .. code-block:: python

            import pennylane as qml
            from pennylane.templates import QAOAEmbedding

            dev = qml.device('default.qubit', wires=2)

            @qml.qnode(dev)
            def circuit(weights, f=None):
                QAOAEmbedding(features=f, weights=weights, wires=range(2))
                return qml.expval(qml.PauliZ(0))

            features = [1., 2.]
            layer1 = [0.1, -0.3, 1.5]
            layer2 = [3.1, 0.2, -2.8]
            weights = [layer1, layer2]

            print(circuit(weights, f=features))

        **Using parameter initialization functions**

        The initial weight parameters can alternatively be generated by utility functions from the
        ``pennylane.init`` module, for example using the function :func:`~.qaoa_embedding_normal`:

        .. code-block:: python

            from pennylane.init import qaoa_embedding_normal
            weights = qaoa_embedding_normal(n_layers=2, n_wires=2, mean=0, std=0.2)


        **Training the embedding**

        The embedding is typically trained with respect to a given cost. For example, one can train it to
        minimize the PauliZ expectation of the first qubit:

        .. code-block:: python

            o = GradientDescentOptimizer()
            for i in range(10):
                weights = o.step(lambda w : circuit(w, f=features), weights)
                print("Step ", i, " weights = ", weights)


        **Training the features**

        In principle, also the features are trainable, which means that gradients with respect to feature values
        can be computed. To train both weights and features, they need to be passed to the qnode as
        positional arguments. If the built-in optimizer is used, they have to be merged to one input:

        .. code-block:: python

            @qml.qnode(dev)
            def circuit2(pars):
                weights = pars[0]
                features = pars[1]
                QAOAEmbedding(features=features, weights=weights, wires=range(2))
                return qml.expval(qml.PauliZ(0))


            features = [1., 2.]
            weights = [[0.1, -0.3, 1.5], [3.1, 0.2, -2.8]]
            pars = [weights, features]

            o = GradientDescentOptimizer()
            for i in range(10):
                pars = o.step(circuit2, pars)
                print("Step ", i, " weights = ", pars[0], " features = ", pars[1])

        **Local Fields**

        While by default, ``RY`` gates are used as local fields, one may also choose ``local_field='Z'`` or
        ``local_field='X'`` as hyperparameters of the embedding.

        .. code-block:: python

            @qml.qnode(dev)
            def circuit(weights, f=None):
                QAOAEmbedding(features=f, weights=weights, wires=range(2), local_field='Z')
                return qml.expval(qml.PauliZ(0))

        Choosing ``'Z'`` fields implements a QAOAEmbedding where the second Hamiltonian is a
        1-dimensional Ising model.

    """
    #############
    # Input checks

    wires = _check_wires(wires)

    expected_shape = (len(wires),)
    _check_shape(features, expected_shape, bound='max', msg="'features' must be of shape {} or smaller; got {}"
                                                            "".format((len(wires),), _get_shape(features)))

    _check_is_in_options(local_field, ['X', 'Y', 'Z'], msg="did not recognize option {} for 'local_field'"
                                                           "".format(local_field))

    repeat = _check_number_of_layers([weights])

    if len(wires) == 1:
        expected_shape = (repeat, 1)
        _check_shape(weights, expected_shape, msg="'weights' must be of shape {}; got {}"
                                                  "".format(expected_shape, _get_shape(features)))
    elif len(wires) == 2:
        expected_shape = (repeat, 3)
        _check_shape(weights, expected_shape, msg="'weights' must be of shape {}; got {}"
                                                  "".format(expected_shape, _get_shape(features)))
    else:
        expected_shape = (repeat, 2*len(wires))
        _check_shape(weights, expected_shape, msg="'weights' must be of shape {}; got {}"
                                                  "".format(expected_shape, _get_shape(features)))

    #####################

    n_features = _get_shape(features)[0]

    if local_field == 'Z':
        local_fields = RZ
    elif local_field == 'X':
        local_fields = RX
    else:
        local_fields = RY

    for l in range(repeat):
        # apply alternating Hamiltonians
        _qaoa_feature_encoding_hamiltonian(features, n_features, wires)
        _qaoa_ising_hamiltonian(weights, wires, local_fields, l)

    # repeat the feature encoding once more at the end
    _qaoa_feature_encoding_hamiltonian(features, n_features, wires)


@template
def DisplacementEmbedding(features, wires, method='amplitude', c=0.1):
    r"""Encodes :math:`N` features into the displacement amplitudes :math:`r` or phases :math:`\phi` of :math:`M` modes,
     where :math:`N\leq M`.

    The mathematical definition of the displacement gate is given by the operator

    .. math::
            D(\alpha) = \exp(r (e^{i\phi}\ad -e^{-i\phi}\a)),

    where :math:`\a` and :math:`\ad` are the bosonic creation and annihilation operators.

    ``features`` has to be an array of at most ``len(wires)`` floats. If there are fewer entries in
    ``features`` than wires, the circuit does not apply the remaining displacement gates.

    Args:
        features (array): Array of features of size (N,)
        wires (Sequence[int]): sequence of mode indices that the template acts on
        method (str): ``'phase'`` encodes the input into the phase of single-mode displacement, while
            ``'amplitude'`` uses the amplitude
        c (float): value of the phase of all displacement gates if ``execution='amplitude'``, or
            the amplitude of all displacement gates if ``execution='phase'``

    Raises:
        ValueError: if inputs do not have the correct format
   """

    #############
    # Input checks

    _check_no_variable(method, msg="'method' cannot be differentiable")
    _check_no_variable(c, msg="'c' cannot be differentiable")

    wires = _check_wires(wires)

    expected_shape = (len(wires),)
    _check_shape(features, expected_shape, bound='max', msg="'features' must be of shape {} or smaller; got {}."
                                                            "".format(expected_shape, _get_shape(features)))

    _check_is_in_options(method, ['amplitude', 'phase'], msg="did not recognize option {} for 'method'"
                                                             "".format(method))

    #############

    for idx, f in enumerate(features):
        if method == 'amplitude':
            Displacement(f, c, wires=wires[idx])
        elif method == 'phase':
            Displacement(c, f, wires=wires[idx])


@template
def SqueezingEmbedding(features, wires, method='amplitude', c=0.1):
    r"""Encodes :math:`N` features into the squeezing amplitudes :math:`r \geq 0` or phases :math:`\phi \in [0, 2\pi)`
    of :math:`M` modes, where :math:`N\leq M`.

    The mathematical definition of the squeezing gate is given by the operator

    .. math::

        S(z) = \exp\left(\frac{r}{2}\left(e^{-i\phi}\a^2 -e^{i\phi}{\ad}^{2} \right) \right),

    where :math:`\a` and :math:`\ad` are the bosonic creation and annihilation operators.

    ``features`` has to be an iterable of at most ``len(wires)`` floats. If there are fewer entries in
    ``features`` than wires, the circuit does not apply the remaining squeezing gates.

    Args:
        features (array): Array of features of size (N,)
        wires (Sequence[int]): sequence of mode indices that the template acts on
        method (str): ``'phase'`` encodes the input into the phase of single-mode squeezing, while
            ``'amplitude'`` uses the amplitude
        c (float): value of the phase of all squeezing gates if ``execution='amplitude'``, or the
            amplitude of all squeezing gates if ``execution='phase'``

    Raises:
        ValueError: if inputs do not have the correct format
    """


    #############
    # Input checks

    _check_no_variable(method, msg="'method' cannot be differentiable")
    _check_no_variable(c, msg="'c' cannot be differentiable")

    wires = _check_wires(wires)

    expected_shape = (len(wires),)
    _check_shape(features, expected_shape, bound='max', msg="'features' must be of shape {} or smaller; got {}"
                                                            "".format(expected_shape, _get_shape(features)))

    _check_is_in_options(method, ['amplitude', 'phase'], msg="did not recognize option {} for 'method'".format(method))

    #############

    for idx, f in enumerate(features):
        if method == 'amplitude':
            Squeezing(f, c, wires=wires[idx])
        elif method == 'phase':
            Squeezing(c, f, wires=wires[idx])


embeddings = {"AngleEmbedding", "AmplitudeEmbedding", "BasisEmbedding", "DisplacementEmbedding",
              "QAOAEmbedding", "SqueezingEmbedding"}

__all__ = list(embeddings)
