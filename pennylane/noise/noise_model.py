# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains class and methods for noise models"""

import inspect

import pennylane as qml


class NoiseModel:
    """Builds a noise model based on a mapping of conditionals to callables that
    defines noise operations using some optional metadata.

    Args:
        model_map (dict[BooleanFn -> Callable]): Data for the
            noise model as a ``{conditional: noise_fn}`` dictionary. The signature of
            ``noise_fn`` must be ``noise_fn(op: Operation, **kwargs) -> None``, where
            ``op`` is the operation that the conditional evaluates and ``kwargs`` are
            the specified metadata arguments.
        **kwargs: Keyword arguments for specifying metadata related to noise model.

    .. note::

        For each key-value pair of ``model_map``:

        - The ``conditional`` should be either a function decorated with :class:`~.BooleanFn`,
          a callable object built via :ref:`constructor functions <intro_boolean_fn>` in
          the ``qml.noise`` module, or their bitwise combination.
        - The definition of ``noise_fn(op, **kwargs)`` should have the operations in the same the order
          in which they are to be queued for an operation ``op``, whenever the corresponding
          ``conditional`` evaluates to ``True``.

    **Example**

    .. code-block:: python

        # Set up the conditionals
        c0 = qml.noise.op_eq(qml.PauliX) | qml.noise.op_eq(qml.PauliY)
        c1 = qml.noise.op_eq(qml.Hadamard) & qml.noise.wires_in([0, 1])

        # Set up the noise functions
        def n0(op, **kwargs):
            qml.ThermalRelaxationError(0.4, kwargs["t1"], 0.2, 0.6, op.wires)

        n1 = qml.noise.partial_wires(qml.AmplitudeDamping, 0.4)

        # Set up noise model
        noise_model = qml.NoiseModel({c0: n0}, t1=0.04)
        noise_model += {c1: n1}

    >>> noise_model
    NoiseModel({
        OpEq(PauliX) | OpEq(PauliY): n0
        OpEq(Hadamard) & WiresIn([0, 1]): AmplitudeDamping(0.4, wires),
    }, t1=0.04)
    """

    def __init__(self, model_map, **kwargs):
        self.check_model(model_map)
        self._model_map = model_map
        self._metadata = kwargs

    @property
    def model_map(self):
        """Gives the conditional model for the noise model."""
        return self._model_map

    @property
    def metadata(self):
        """Gives the metadata for the noise model."""
        return self._metadata

    def __add__(self, data):
        if not isinstance(data, NoiseModel):
            return NoiseModel({**self.model_map, **data}, **self.metadata)

        return NoiseModel(
            {**self.model_map, **data.model_map}, **{**self.metadata, **data.metadata}
        )

    def __radd__(self, data):
        return self.__add__(data)

    def __sub__(self, data):
        if not isinstance(data, NoiseModel):
            return NoiseModel(
                {k: v for k, v in self.model_map.items() if k not in data}, **self.metadata
            )

        return NoiseModel(
            {k: v for k, v in self.model_map.items() if k not in data.model_map},
            **dict({k: v for k, v in self.metadata.items() if k not in data.metadata}),
        )

    def __eq__(self, other):
        for model1, model2 in zip(self.model_map.items(), other.model_map.items()):
            (fcond1, noise1), (fcond2, noise2) = model1, model2
            if getattr(fcond1, "condition", fcond1.fn) != getattr(fcond2, "condition", fcond2.fn):
                return False
            if noise1 != noise2:
                return False
        return self.metadata == other.metadata

    def __repr__(self):
        model_str = "NoiseModel({\n"
        for key, val in self.model_map.items():
            model_str += "    " + f"{key}: {val.__name__}" + "\n"
        model_str += "}, "
        for key, val in self._metadata.items():
            model_str += f"{key} = {val}, "
        model_str = model_str[:-2] + ")"

        return model_str

    @staticmethod
    def check_model(model: dict) -> None:
        """Method to validate the ``model`` map for constructing a NoiseModel."""
        for condition, noise in model.items():
            if not isinstance(condition, qml.BooleanFn):
                raise ValueError(
                    f"{condition} must be a boolean conditional, i.e., an instance of "
                    "BooleanFn or one of its subclasses."
                )

            final_parameter = list(inspect.signature(noise).parameters.values())[-1]
            if final_parameter.kind != final_parameter.VAR_KEYWORD:
                raise ValueError(
                    f"{noise} provided for {condition} must accept **kwargs "
                    "as the last argument in its signature."
                )
