# Copyright 2018-2021 Xanadu Quantum Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains the classes and functions for integrating QNodes with the Keras Layer
API."""
import inspect
from collections.abc import Iterable
from typing import Optional, Text
from semantic_version import Version


try:
    import tensorflow as tf
    from tensorflow.keras.layers import Layer

    CORRECT_TF_VERSION = Version(tf.__version__) >= Version("2.0.0")
    try:
        # this feels a bit hacky, but if users *only* have an old (i.e. PL-compatible) version of Keras installed
        # then tf.keras doesn't have a version attribute, and we *should be* good to go.
        # if you have a newer version of Keras installed, then you can use tf.keras.version to check if you
        # are configured to use Keras 3 or Keras 2
        CORRECT_KERAS_VERSION = Version(tf.keras.version()) < Version("3.0.0")
    except AttributeError:
        CORRECT_KERAS_VERSION = True

except ImportError:
    # The following allows this module to be imported even if TensorFlow is not installed. Users
    # will instead see an ImportError when instantiating the KerasLayer.
    from abc import ABC

    Layer = ABC
    CORRECT_TF_VERSION = False


class KerasLayer(Layer):
    """Converts a :class:`~.QNode` to a Keras
    `Layer <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer>`__.

    The result can be used within the Keras
    `Sequential <https://www.tensorflow.org/api_docs/python/tf/keras/Sequential>`__ or
    `Model <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`__ classes for
    creating quantum and hybrid models.

    .. note::

        ``KerasLayer`` currently only supports Keras 2. If you are running the newest version
        of TensorFlow and Keras, you may automatically be using Keras 3. For instructions
        on running with Keras 2, instead, see the
        `documentation on backwards compatibility <https://keras.io/getting_started/#tensorflow--keras-2-backwards-compatibility>`__ .

    Args:
        qnode (qml.QNode): the PennyLane QNode to be converted into a Keras Layer_
        weight_shapes (dict[str, tuple]): a dictionary mapping from all weights used in the QNode to
            their corresponding shapes
        output_dim (int): the output dimension of the QNode
        weight_specs (dict[str, dict]): An optional dictionary for users to provide additional
            specifications for weights used in the QNode, such as the method of parameter
            initialization. This specification is provided as a dictionary with keys given by the
            arguments of the `add_weight()
            <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#add_weight>`__
            method and values being the corresponding specification.
        **kwargs: additional keyword arguments passed to the Layer_ base class

    **Example**

    First let's define the QNode that we want to convert into a Keras Layer_:

    .. code-block:: python

        n_qubits = 2
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def qnode(inputs, weights_0, weight_1):
            qml.RX(inputs[0], wires=0)
            qml.RX(inputs[1], wires=1)
            qml.Rot(*weights_0, wires=0)
            qml.RY(weight_1, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

    The signature of the QNode **must** contain an ``inputs`` named argument for input data,
    with all other arguments to be treated as internal weights. We can then convert to a Keras
    Layer_ with:

    >>> weight_shapes = {"weights_0": 3, "weight_1": 1}
    >>> qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=2)

    The internal weights of the QNode are automatically initialized within the
    :class:`~.KerasLayer` and must have their shapes specified in a ``weight_shapes`` dictionary.
    It is then easy to combine with other neural network layers from the
    `tensorflow.keras.layers <https://www.tensorflow.org/api_docs/python/tf/keras/layers>`__ module
    and create a hybrid:

    >>> clayer = tf.keras.layers.Dense(2)
    >>> model = tf.keras.models.Sequential([qlayer, clayer])

    .. details::
        :title: Usage Details

        **QNode signature**

        The QNode must have a signature that satisfies the following conditions:

        - Contain an ``inputs`` named argument for input data.
        - All other arguments must accept an array or tensor and are treated as internal
          weights of the QNode.
        - All other arguments must have no default value.
        - The ``inputs`` argument is permitted to have a default value provided the gradient with
          respect to ``inputs`` is not required.
        - There cannot be a variable number of positional or keyword arguments, e.g., no ``*args``
          or ``**kwargs`` present in the signature.

        **Output shape**

        If the QNode returns a single measurement, then the output of the ``KerasLayer`` will have
        shape ``(batch_dim, *measurement_shape)``, where ``measurement_shape`` is the output shape
        of the measurement:

        .. code-block::

            def print_output_shape(measurements):
                n_qubits = 2
                dev = qml.device("default.qubit", wires=n_qubits, shots=100)

                @qml.qnode(dev)
                def qnode(inputs, weights):
                    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
                    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                    if len(measurements) == 1:
                        return qml.apply(measurements[0])
                    return [qml.apply(m) for m in measurements]

                weight_shapes = {"weights": (3, n_qubits, 3)}
                qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=None)

                batch_dim = 5
                x = tf.zeros((batch_dim, n_qubits))
                return qlayer(x).shape

        >>> print_output_shape([qml.expval(qml.Z(0))])
        TensorShape([5])
        >>> print_output_shape([qml.probs(wires=[0, 1])])
        TensorShape([5, 4])
        >>> print_output_shape([qml.sample(wires=[0, 1])])
        TensorShape([5, 100, 2])

        If the QNode returns multiple measurements, then the measurement results will be flattened
        and concatenated, resulting in an output of shape ``(batch_dim, total_flattened_dim)``:

        >>> print_output_shape([qml.expval(qml.Z(0)), qml.probs(wires=[0, 1])])
        TensorShape([5, 5])
        >>> print_output_shape([qml.probs([0, 1]), qml.sample(wires=[0, 1])])
        TensorShape([5, 204])

        **Initializing weights**

        The optional ``weight_specs`` argument of :class:`~.KerasLayer` allows for a more
        fine-grained specification of the QNode weights, such as the method of initialization and
        any regularization or constraints. For example, the initialization method of the ``weights``
        argument in the example above could be specified by:

        .. code-block::

            weight_specs = {"weights": {"initializer": "random_uniform"}}

        The values of ``weight_specs`` are dictionaries with keys given by arguments of
        the Keras
        `add_weight() <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#add_weight>`__
        method. For the ``"initializer"`` argument, one can specify a string such as
        ``"random_uniform"`` or an instance of an `Initializer
        <https://www.tensorflow.org/api_docs/python/tf/keras/initializers>`__ class, such as
        `tf.keras.initializers.RandomUniform <https://www.tensorflow.org/api_docs/python/tf/random_uniform_initializer>`__.

        If ``weight_specs`` is not specified, weights will be added using the Keras default
        initialization and without any regularization or constraints.

        **Model saving**

        The weights of models that contain ``KerasLayers`` can be saved using the usual
        ``tf.keras.Model.save_weights`` method:

        .. code-block::

            clayer = tf.keras.layers.Dense(2, input_shape=(2,))
            qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=2)
            model = tf.keras.Sequential([clayer, qlayer])
            model.save_weights(SAVE_PATH)

        To load the model weights, first instantiate the model like before, then call
        ``tf.keras.Model.load_weights``:

        .. code-block::

            clayer = tf.keras.layers.Dense(2, input_shape=(2,))
            qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=2)
            model = tf.keras.Sequential([clayer, qlayer])
            model.load_weights(SAVE_PATH)

        Models containing ``KerasLayer`` objects can also be saved directly using
        ``tf.keras.Model.save``. This method also saves the model architecture, weights,
        and training configuration, including the optimizer state:

        .. code-block::

            clayer = tf.keras.layers.Dense(2, input_shape=(2,))
            qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=2)
            model = tf.keras.Sequential([clayer, qlayer])
            model.save(SAVE_PATH)

        In this case, loading the model requires no knowledge of the original source code:

        .. code-block::

            model = tf.keras.models.load_model(SAVE_PATH)

        .. note::

            Currently ``KerasLayer`` objects cannot be saved in the ``HDF5`` file format. In order
            to save a model using the latter method above, the ``SavedModel`` file format (default
            in TensorFlow 2.x) should be used.

        **Additional example**

        The code block below shows how a circuit composed of templates from the
        :doc:`/introduction/templates` module can be combined with classical
        `Dense <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense>`__ layers to learn
        the two-dimensional `moons <https://scikit-learn.org/stable/modules/generated/sklearn
        .datasets.make_moons.html>`__ dataset.

        .. code-block:: python

            import pennylane as qml
            import tensorflow as tf
            import sklearn.datasets

            n_qubits = 2
            dev = qml.device("default.qubit", wires=n_qubits)

            @qml.qnode(dev)
            def qnode(inputs, weights):
                qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
                qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

            weight_shapes = {"weights": (3, n_qubits, 3)}

            qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=2)
            clayer1 = tf.keras.layers.Dense(2)
            clayer2 = tf.keras.layers.Dense(2, activation="softmax")
            model = tf.keras.models.Sequential([clayer1, qlayer, clayer2])

            data = sklearn.datasets.make_moons()
            X = tf.constant(data[0])
            Y = tf.one_hot(data[1], depth=2)

            opt = tf.keras.optimizers.SGD(learning_rate=0.5)
            model.compile(opt, loss='mae')

        The model can be trained using:

        >>> model.fit(X, Y, epochs=8, batch_size=5)
        Train on 100 samples
        Epoch 1/8
        100/100 [==============================] - 9s 90ms/sample - loss: 0.3524
        Epoch 2/8
        100/100 [==============================] - 9s 87ms/sample - loss: 0.2441
        Epoch 3/8
        100/100 [==============================] - 9s 87ms/sample - loss: 0.1908
        Epoch 4/8
        100/100 [==============================] - 9s 87ms/sample - loss: 0.1832
        Epoch 5/8
        100/100 [==============================] - 9s 88ms/sample - loss: 0.1596
        Epoch 6/8
        100/100 [==============================] - 9s 87ms/sample - loss: 0.1637
        Epoch 7/8
        100/100 [==============================] - 9s 86ms/sample - loss: 0.1613
        Epoch 8/8
        100/100 [==============================] - 9s 87ms/sample - loss: 0.1474

        **Returning a state**

        If your QNode returns the state of the quantum circuit using :func:`~pennylane.state` or
        :func:`~pennylane.density_matrix`, you must immediately follow your quantum Keras Layer with a layer
        that casts to reals. For example, you could use
        `tf.keras.layers.Lambda <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Lambda>`__
        with the function ``lambda x: tf.abs(x)``. This casting is required because TensorFlow's
        Keras layers require a real input and are differentiated with respect to real parameters.

    .. _Layer: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer
    """

    def __init__(
        self,
        qnode,
        weight_shapes: dict,
        output_dim,
        weight_specs: Optional[dict] = None,
        **kwargs,
    ):
        # pylint: disable=too-many-arguments
        if not CORRECT_TF_VERSION:
            raise ImportError(
                "KerasLayer requires TensorFlow version 2 or above. The latest "
                "version of TensorFlow can be installed using:\n"
                "pip install tensorflow --upgrade\nAlternatively, visit "
                "https://www.tensorflow.org/install for detailed instructions."
            )

        if not CORRECT_KERAS_VERSION:
            raise ImportError(
                "KerasLayer requires a Keras version lower than 3. For instructions on running with Keras 2,"
                "visit https://keras.io/getting_started/#tensorflow--keras-2-backwards-compatibility."
            )

        self.weight_shapes = {
            weight: (tuple(size) if isinstance(size, Iterable) else (size,) if size > 1 else ())
            for weight, size in weight_shapes.items()
        }

        self._signature_validation(qnode, weight_shapes)

        self.qnode = qnode
        self.qnode.interface = "tf"

        # Allows output_dim to be specified as an int or as a tuple, e.g, 5, (5,), (5, 2), [5, 2]
        # Note: Single digit values will be considered an int and multiple as a tuple, e.g [5,] or (5,)
        # are passed as integer 5 and [5, 2] will be passes as tuple (5, 2)
        if isinstance(output_dim, Iterable) and len(output_dim) > 1:
            self.output_dim = tuple(output_dim)
        else:
            self.output_dim = output_dim[0] if isinstance(output_dim, Iterable) else output_dim

        self.weight_specs = weight_specs if weight_specs is not None else {}

        self.qnode_weights = {}

        super().__init__(dynamic=True, **kwargs)

        # no point in delaying the initialization of weights, since we already know their shapes
        self.build(None)
        self._initialized = True

    def _signature_validation(self, qnode, weight_shapes):
        sig = inspect.signature(qnode.func).parameters

        if self.input_arg not in sig:
            raise TypeError(
                f"QNode must include an argument with name {self.input_arg} for inputting data"
            )

        if self.input_arg in set(weight_shapes.keys()):
            raise ValueError(
                f"{self.input_arg} argument should not have its dimension specified in "
                f"weight_shapes"
            )

        param_kinds = [p.kind for p in sig.values()]

        if inspect.Parameter.VAR_POSITIONAL in param_kinds:
            raise TypeError("Cannot have a variable number of positional arguments")

        if inspect.Parameter.VAR_KEYWORD not in param_kinds:
            if set(weight_shapes.keys()) | {self.input_arg} != set(sig.keys()):
                raise ValueError("Must specify a shape for every non-input parameter in the QNode")

    def build(self, input_shape):
        """Initializes the QNode weights.

        Args:
            input_shape (tuple or tf.TensorShape): shape of input data; this is unused since
                the weight shapes are already known in the __init__ method.
        """
        for weight, size in self.weight_shapes.items():
            spec = self.weight_specs.get(weight, {})
            self.qnode_weights[weight] = self.add_weight(name=weight, shape=size, **spec)

        super().build(input_shape)

    def call(self, inputs):
        """Evaluates the QNode on input data using the initialized weights.

        Args:
            inputs (tensor): data to be processed

        Returns:
            tensor: output data
        """
        has_batch_dim = len(inputs.shape) > 1

        # in case the input has more than one batch dimension
        if has_batch_dim:
            batch_dims = tf.shape(inputs)[:-1]
            inputs = tf.reshape(inputs, (-1, inputs.shape[-1]))

        # calculate the forward pass as usual
        results = self._evaluate_qnode(inputs)

        # reshape to the correct number of batch dims
        if has_batch_dim:
            # pylint:disable=unexpected-keyword-arg,no-value-for-parameter
            new_shape = tf.concat([batch_dims, tf.shape(results)[1:]], axis=0)
            results = tf.reshape(results, new_shape)

        return results

    def _evaluate_qnode(self, x):
        """Evaluates a QNode for a single input datapoint.

        Args:
            x (tensor): the datapoint

        Returns:
            tensor: output datapoint
        """
        kwargs = {
            **{self.input_arg: x},
            **{k: 1.0 * w for k, w in self.qnode_weights.items()},
        }
        res = self.qnode(**kwargs)

        if isinstance(res, (list, tuple)):
            if len(x.shape) > 1:
                # multi-return and batch dim case
                res = [tf.reshape(r, (tf.shape(x)[0], tf.reduce_prod(r.shape[1:]))) for r in res]

            # multi-return and no batch dim
            return tf.experimental.numpy.hstack(res)

        return res

    def construct(self, args, kwargs):
        """Constructs the wrapped QNode on input data using the initialized weights.

        This method was added to match the QNode interface. The provided args
        must contain a single item, which is the input to the layer. The provided
        kwargs is unused.

        Args:
            args (tuple): A tuple containing one entry that is the input to this layer
            kwargs (dict): Unused
        """
        # GradientTape required to ensure that the weights show up as trainable on the qtape
        with tf.GradientTape() as tape:
            tape.watch(list(self.qnode_weights.values()))

            inputs = args[0]
            kwargs = {self.input_arg: inputs, **{k: 1.0 * w for k, w in self.qnode_weights.items()}}
            self.qnode.construct((), kwargs)

    def __getattr__(self, item):
        """If the given attribute does not exist in the class, look for it in the wrapped QNode."""
        if self._initialized and hasattr(self.qnode, item):
            return getattr(self.qnode, item)

        return super().__getattr__(item)

    def __setattr__(self, item, val):
        """If the given attribute does not exist in the class, try to set it in the wrapped QNode."""
        if self._initialized and hasattr(self.qnode, item):
            setattr(self.qnode, item, val)
        else:
            super().__setattr__(item, val)

    def compute_output_shape(self, input_shape):
        """Computes the output shape after passing data of shape ``input_shape`` through the
        QNode.

        Args:
            input_shape (tuple or tf.TensorShape): shape of input data

        Returns:
            tf.TensorShape: shape of output data
        """
        return tf.TensorShape([input_shape[0]]).concatenate(self.output_dim)

    def __str__(self):
        detail = "<Quantum Keras Layer: func={}>"
        return detail.format(self.qnode.func.__name__)

    __repr__ = __str__

    _input_arg = "inputs"
    _initialized = False

    @property
    def input_arg(self):
        """Name of the argument to be used as the input to the Keras
        `Layer <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer>`__. Set to
        ``"inputs"``."""
        return self._input_arg

    @staticmethod
    def set_input_argument(input_name: Text = "inputs") -> None:
        """
        Set the name of the input argument.

        Args:
            input_name (str): Name of the input argument
        """
        KerasLayer._input_arg = input_name

    def get_config(self) -> dict:
        """
        Get serialized layer configuration

        Returns:
            dict: layer configuration
        """
        config = super().get_config()

        config.update(
            {
                "output_dim": self.output_dim,
                "weight_specs": self.weight_specs,
                "weight_shapes": self.weight_shapes,
            }
        )
        return config
