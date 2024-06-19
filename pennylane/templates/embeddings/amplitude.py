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
r"""
Contains the AmplitudeEmbedding template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import pennylane as qml
from pennylane.ops import StatePrep
from pennylane.wires import Wires

# tolerance for normalization
TOLERANCE = 1e-10


class AmplitudeEmbedding(StatePrep):
    r"""Encodes :math:`2^n` features into the amplitude vector of :math:`n` qubits.

    By setting ``pad_with`` to a real or complex number, ``features`` is automatically padded to dimension
    :math:`2^n` where :math:`n` is the number of qubits used in the embedding.

    To represent a valid quantum state vector, the L2-norm of ``features`` must be one.
    The argument ``normalize`` can be set to ``True`` to automatically normalize the features.

    If both automatic padding and normalization are used, padding is executed *before* normalizing.

    .. note::

        On some devices, ``AmplitudeEmbedding`` must be the first operation of a quantum circuit.

    .. warning::

        At the moment, the ``features`` argument is **not differentiable** when using the template, and
        gradients with respect to the features cannot be computed by PennyLane.

    Args:
        features (tensor_like): input tensor of dimension ``(2^len(wires),)``, or less if `pad_with` is specified
        wires (Any or Iterable[Any]): wires that the template acts on
        pad_with (float or complex): if not None, the input is padded with this constant to size :math:`2^n`
        normalize (bool): whether to automatically normalize the features
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified.

    Example:

        Amplitude embedding encodes a normalized :math:`2^n`-dimensional feature vector into the state
        of :math:`n` qubits:

        .. code-block:: python

            import pennylane as qml

            dev = qml.device('default.qubit', wires=2)

            @qml.qnode(dev)
            def circuit(f=None):
                qml.AmplitudeEmbedding(features=f, wires=range(2))
                return qml.expval(qml.Z(0)), qml.state()

            res, state = circuit(f=[1/2, 1/2, 1/2, 1/2])

        The final state of the device is - up to a global phase - equivalent to the input passed to the circuit:

        >>> state
        tensor([0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j], requires_grad=True)

        **Differentiating with respect to the features**

        Due to non-trivial classical processing to construct the state preparation circuit,
        the features argument is in general **not differentiable**.

        **Normalization**

        The template will raise an error if the feature input is not normalized.
        One can set ``normalize=True`` to automatically normalize it:

        .. code-block:: python

            @qml.qnode(dev)
            def circuit(f=None):
                qml.AmplitudeEmbedding(features=f, wires=range(2), normalize=True)
                return qml.expval(qml.Z(0)), qml.state()

            res, state = circuit(f=[15, 15, 15, 15])

        >>> state
        tensor([0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j], requires_grad=True)

        **Padding**

        If the dimension of the feature vector is smaller than the number of amplitudes,
        one can automatically pad it with a constant for the missing dimensions using the ``pad_with`` option:

        .. code-block:: python

            from math import sqrt

            @qml.qnode(dev)
            def circuit(f=None):
                qml.AmplitudeEmbedding(features=f, wires=range(2), pad_with=0.)
                return qml.expval(qml.Z(0)), qml.state()

            res, state = circuit(f=[1/sqrt(2), 1/sqrt(2)])

        >>> state
        tensor([0.70710678+0.j, 0.70710678+0.j, 0.        +0.j, 0.        +0.j], requires_grad=True)

    """

    def __init__(self, features, wires, pad_with=None, normalize=False, id=None):
        # pylint:disable=bad-super-call
        wires = Wires(wires)
        self.pad_with = pad_with
        self.normalize = normalize
        features = self._preprocess(features, wires, pad_with, normalize)
        super(StatePrep, self).__init__(features, wires=wires, id=id)

    @staticmethod
    def compute_decomposition(
        features, wires
    ):  # pylint: disable=arguments-differ,arguments-renamed
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.AmplitudeEmbedding.decomposition`.

        Args:
            features (tensor_like): input tensor of dimension ``(2^len(wires),)``
            wires (Any or Iterable[Any]): wires that the operator acts on

        Returns:
            list[.Operator]: decomposition of the operator

        **Example**

        >>> features = torch.tensor([1., 0., 0., 0.])
        >>> qml.AmplitudeEmbedding.compute_decomposition(features, wires=["a", "b"])
        [StatePrep(tensor([1., 0., 0., 0.]), wires=['a', 'b'])]
        """
        return [StatePrep(features, wires=wires)]

    @staticmethod
    def _preprocess(features, wires, pad_with, normalize):
        """Validate and pre-process inputs as follows:

        * If features is batched, the processing that follows is applied to each feature set in the batch.
        * Check that the features tensor is one-dimensional.
        * If pad_with is None, check that the last dimension of the features tensor
          has length :math:`2^n` where :math:`n` is the number of qubits. Else check that the
          last dimension of the features tensor is not larger than :math:`2^n` and pad features
          with value if necessary.
        * If normalize is false, check that last dimension of features is normalised to one. Else, normalise the
          features tensor.
        """
        shape = qml.math.shape(features)

        # check shape
        if len(shape) not in (1, 2):
            raise ValueError(
                f"Features must be a one-dimensional tensor, or two-dimensional with batching; got shape {shape}."
            )

        n_features = shape[-1]
        dim = 2 ** len(wires)
        if pad_with is None and n_features != dim:
            raise ValueError(
                f"Features must be of length {dim}; got length {n_features}. "
                f"Use the 'pad_with' argument for automated padding."
            )

        if pad_with is not None:
            if n_features > dim:
                raise ValueError(
                    f"Features must be of length {dim} or "
                    f"smaller to be padded; got length {n_features}."
                )

            # pad
            if n_features < dim:
                padding = [pad_with] * (dim - n_features)
                if len(shape) > 1:
                    padding = [padding] * shape[0]
                padding = qml.math.convert_like(padding, features)
                features = qml.math.hstack([features, padding])

        # normalize
        norm = qml.math.sum(qml.math.abs(features) ** 2, axis=-1)

        if qml.math.is_abstract(norm):
            if normalize or pad_with:
                features = features / qml.math.reshape(qml.math.sqrt(norm), (*shape[:-1], 1))

        elif not qml.math.allclose(norm, 1.0, atol=TOLERANCE):
            if normalize or pad_with:
                features = features / qml.math.reshape(qml.math.sqrt(norm), (*shape[:-1], 1))
            else:
                raise ValueError(
                    f"Features must be a vector of norm 1.0; got norm {norm}. "
                    "Use 'normalize=True' to automatically normalize."
                )

        return features
