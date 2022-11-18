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
# pylint: disable=protected-access
"""
This module contains the functions for computing different types of measurement
outcomes from quantum observables - expectation values, variances of expectations,
and measurement samples using AnnotatedQueues.
"""
# pylint: disable=too-many-instance-attributes
import contextlib
import copy
import functools
from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence, Tuple, Union

import numpy as np

import pennylane as qml
from pennylane.operation import Observable
from pennylane.wires import Wires

# =============================================================================
# ObservableReturnTypes types
# =============================================================================


class ObservableReturnTypes(Enum):
    """Enumeration class to represent the return types of an observable."""

    Sample = "sample"
    Counts = "counts"
    AllCounts = "allcounts"
    Variance = "var"
    Expectation = "expval"
    Probability = "probs"
    State = "state"
    MidMeasure = "measure"
    VnEntropy = "vnentropy"
    MutualInfo = "mutualinfo"
    Shadow = "shadow"
    ShadowExpval = "shadowexpval"

    def __repr__(self):
        """String representation of the return types."""
        return str(self.value)


Sample = ObservableReturnTypes.Sample
"""Enum: An enumeration which represents sampling an observable."""

Counts = ObservableReturnTypes.Counts
"""Enum: An enumeration which represents returning the number of times
 each of the observed outcomes occurred in sampling."""

AllCounts = ObservableReturnTypes.AllCounts
"""Enum: An enumeration which represents returning the number of times
 each of the possible outcomes occurred in sampling, including 0 counts
 for unobserved outcomes."""

Variance = ObservableReturnTypes.Variance
"""Enum: An enumeration which represents returning the variance of
an observable on specified wires."""

Expectation = ObservableReturnTypes.Expectation
"""Enum: An enumeration which represents returning the expectation
value of an observable on specified wires."""

Probability = ObservableReturnTypes.Probability
"""Enum: An enumeration which represents returning probabilities
of all computational basis states."""

State = ObservableReturnTypes.State
"""Enum: An enumeration which represents returning the state in the computational basis."""

MidMeasure = ObservableReturnTypes.MidMeasure
"""Enum: An enumeration which represents returning sampling the computational
basis in the middle of the circuit."""

VnEntropy = ObservableReturnTypes.VnEntropy
"""Enum: An enumeration which represents returning Von Neumann entropy before measurements."""

MutualInfo = ObservableReturnTypes.MutualInfo
"""Enum: An enumeration which represents returning the mutual information before measurements."""

Shadow = ObservableReturnTypes.Shadow
"""Enum: An enumeration which represents returning the bitstrings and recipes from
the classical shadow protocol"""

ShadowExpval = ObservableReturnTypes.ShadowExpval
"""Enum: An enumeration which represents returning the estimated expectation value
from a classical shadow measurement"""


class MeasurementShapeError(ValueError):
    """An error raised when an unsupported operation is attempted with a
    quantum tape."""


class MeasurementProcess(ABC):
    """Represents a measurement process occurring at the end of a
    quantum variational circuit.

    Args:
        obs (.Observable): The observable that is to be measured as part of the
            measurement process. Not all measurement processes require observables (for
            example ``Probability``); this argument is optional.
        wires (.Wires): The wires the measurement process applies to.
            This can only be specified if an observable was not provided.
        eigvals (array): A flat array representing the eigenvalues of the measurement.
            This can only be specified if an observable was not provided.
        id (str): custom label given to a measurement instance, can be useful for some applications
            where the instance has to be identified
        log_base (float): Base for the logarithm.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        obs: Union[Observable, None] = None,
        wires=None,
        eigvals=None,
        id=None,
    ):
        self.obs = obs
        self.id = id

        if wires is not None:
            if len(wires) == 0:
                raise ValueError("Cannot set an empty list of wires.")
            if obs is not None:
                raise ValueError("Cannot set the wires if an observable is provided.")

        # _wires = None indicates broadcasting across all available wires.
        # It translates to the public property wires = Wires([])
        self._wires = wires
        self._eigvals = None

        if eigvals is not None:
            if obs is not None:
                raise ValueError("Cannot set the eigenvalues if an observable is provided.")

            self._eigvals = np.array(eigvals)

        # TODO: remove the following lines once devices
        # have been refactored to accept and understand receiving
        # measurement processes rather than specific observables.

        # The following lines are only applicable for measurement processes
        # that do not have corresponding observables (e.g., Probability). We use
        # them to 'trick' the device into thinking it has received an observable.

        # Below, we imitate an identity observable, so that the
        # device undertakes no action upon receiving this observable.
        self.name = "Identity"
        self.data = []

        # Queue the measurement process
        self.queue()

    @property
    def return_type(self):
        """Measurement return type."""
        return None

    @property
    def numeric_type(self):
        """The Python numeric type of the measurement result.

        Returns:
            type: The output numeric type; ``int``, ``float`` or ``complex``.

        Raises:
            QuantumFunctionError: the return type of the measurement process is
                unrecognized and cannot deduce the numeric type
        """
        raise qml.QuantumFunctionError(
            "Cannot deduce the numeric type of the measurement process with unrecognized "
            + f"return_type {self.return_type}."
        )

    def shape(self, device=None):
        """The expected output shape of the MeasurementProcess.

        Note that the output shape is dependent on the device when:

        * The ``return_type`` is either ``Probability``, ``State`` (from :func:`.state`) or
          ``Sample``;
        * The shot vector was defined in the device.

        For example, assuming a device with ``shots=None``, expectation values
        and variances define ``shape=(1,)``, whereas probabilities in the qubit
        model define ``shape=(1, 2**num_wires)`` where ``num_wires`` is the
        number of wires the measurement acts on.

        Note that the shapes for vector-valued return types such as
        ``Probability`` and ``State`` are adjusted to the output of
        ``qml.execute`` and may have an extra first element that is squeezed
        when using QNodes.

        Args:
            device (.Device): a PennyLane device to use for determining the
                shape

        Returns:
            tuple: the output shape

        Raises:
            QuantumFunctionError: the return type of the measurement process is
                unrecognized and cannot deduce the numeric type
        """
        if qml.active_return():
            return self._shape_new(device=device)
        raise qml.QuantumFunctionError(
            "Cannot deduce the shape of the measurement process with unrecognized return_type "
            + f"{self.return_type}."
        )

    def _shape_new(self, device=None):
        """The expected output shape of the MeasurementProcess.

        Note that the output shape is dependent on the device when:

        * The ``return_type`` is either ``Probability``, ``State`` (from :func:`.state`) or
          ``Sample``;
        * The shot vector was defined in the device.

        For example, assuming a device with ``shots=None``, expectation values
        and variances define ``shape=(,)``, whereas probabilities in the qubit
        model define ``shape=(2**num_wires)`` where ``num_wires`` is the
        number of wires the measurement acts on.

        Args:
            device (.Device): a PennyLane device to use for determining the
                shape

        Returns:
            tuple: the output shape

        Raises:
            QuantumFunctionError: the return type of the measurement process is
                unrecognized and cannot deduce the numeric type
        """
        raise qml.QuantumFunctionError(
            "Cannot deduce the shape of the measurement process with unrecognized return_type "
            + f"{self.return_type}."
        )

    @staticmethod
    @functools.lru_cache()
    def _get_num_basis_states(num_wires, device):
        """Auxiliary function to determine the number of basis states given the
        number of systems and a quantum device.

        This function is meant to be used with the Probability measurement to
        determine how many outcomes there will be. With qubit based devices
        we'll have two outcomes for each subsystem. With continuous variable
        devices that impose a Fock cutoff the number of basis states per
        subsystem equals the cutoff value.

        Args:
            num_wires (int): the number of qubits/qumodes
            device (.Device): a PennyLane device

        Returns:
            int: the number of basis states
        """
        cutoff = getattr(device, "cutoff", None)
        base = 2 if cutoff is None else cutoff
        return base**num_wires

    def diagonalizing_gates(self):
        """Returns the gates that diagonalize the measured wires such that they
        are in the eigenbasis of the circuit observables.

        Returns:
            List[.Operation]: the operations that diagonalize the observables
        """
        try:
            # pylint: disable=no-member
            return self.expand().operations
        except qml.operation.DecompositionUndefinedError:
            return []

    def __repr__(self):
        """Representation of this class."""
        if self.obs is None:
            return f"{self.return_type.value}(wires={self.wires.tolist()})"

        # Todo: when tape is core the return type will always be taken from the MeasurementProcess
        if self.obs.return_type is None:
            return f"{self.return_type.value}({self.obs})"

        return f"{self.obs}"

    def __copy__(self):
        return self.__class__(
            obs=copy.copy(self.obs),
            wires=self._wires,
            eigvals=self._eigvals,
        )

    @property
    def wires(self):
        r"""The wires the measurement process acts on.

        This is the union of all the Wires objects of the measurement.
        """
        if self.obs is not None:
            return self.obs.wires

        return (
            Wires.all_wires(self._wires)
            if isinstance(self._wires, list)
            else self._wires or Wires([])
        )

    @property
    def raw_wires(self):
        r"""The wires the measurement process acts on.

        For measurements involving more than one set of wires (such as
        mutual information), this is a list of the Wires objects. Otherwise,
        this is the same as :func:`~.MeasurementProcess.wires`
        """
        return self._wires

    def eigvals(self):
        r"""Eigenvalues associated with the measurement process.

        If the measurement process has an associated observable,
        the eigenvalues will correspond to this observable. Otherwise,
        they will be the eigenvalues provided when the measurement
        process was instantiated.

        Note that the eigenvalues are not guaranteed to be in any
        particular order.

        **Example:**

        >>> m = MeasurementProcess(Expectation, obs=qml.PauliX(wires=1))
        >>> m.eigvals()
        array([1, -1])

        Returns:
            array: eigvals representation
        """
        if self.obs is not None:
            with contextlib.suppress(qml.operation.EigvalsUndefinedError):
                return self.obs.eigvals()
        return self._eigvals

    @property
    def has_decomposition(self):
        r"""Bool: Whether or not the MeasurementProcess returns a defined decomposition
        when calling ``expand``.
        """
        # If self.obs is not None, `expand` queues the diagonalizing gates of self.obs,
        # which we have to check to be defined. The subsequent creation of the new
        # `MeasurementProcess` within `expand` should never fail with the given parameters.
        return False if self.obs is None else self.obs.has_diagonalizing_gates

    @property
    def samples_computational_basis(self):
        r"""Bool: Whether or not the MeasurementProcess returns samples in the computational basis or counts of
        computational basis states.
        """
        return False

    def expand(self):
        """Expand the measurement of an observable to a unitary
        rotation and a measurement in the computational basis.

        Returns:
            .QuantumScript: a quantum script containing the operations
            required to diagonalize the observable

        **Example**

        Consider a measurement process consisting of the expectation
        value of an Hermitian observable:

        >>> H = np.array([[1, 2], [2, 4]])
        >>> obs = qml.Hermitian(H, wires=['a'])
        >>> m = MeasurementProcess(Expectation, obs=obs)

        Expanding this out:

        >>> qscript = m.expand()

        We can see that the resulting script has the qubit unitary applied,
        and a measurement process with no observable, but the eigenvalues
        specified:

        >>> print(qscript.operations)
        [QubitUnitary(array([[-0.89442719,  0.4472136 ],
              [ 0.4472136 ,  0.89442719]]), wires=['a'])]
        >>> print(qscript.measurements[0].eigvals())
        [0. 5.]
        >>> print(qscript.measurements[0].obs)
        None
        """
        if self.obs is None:
            raise qml.operation.DecompositionUndefinedError

        with qml.queuing.AnnotatedQueue() as q:
            self.obs.diagonalizing_gates()
            self.__class__(wires=self.obs.wires, eigvals=self.obs.eigvals())

        return qml.tape.QuantumScript.from_queue(q)

    def queue(self, context=qml.QueuingManager):
        """Append the measurement process to an annotated queue."""
        if self.obs is not None:
            context.update_info(self.obs, owner=self)
            context.append(self, owns=self.obs)
        else:
            context.append(self)

        return self

    @property
    def _queue_category(self):
        """Denotes that `MeasurementProcess` objects should be processed into the `_measurements` list
        in `QuantumTape` objects.

        This property is a temporary solution that should not exist long-term and should not be
        used outside of ``QuantumTape._process_queue``.
        """
        return "_measurements"

    @property
    def hash(self):
        """int: returns an integer hash uniquely representing the measurement process"""

        if self.obs is None:
            fingerprint = (
                str(self.name),
                tuple(self.wires.tolist()),
                str(self.data),
                self.return_type,
            )
        else:
            fingerprint = (
                str(self.obs.name),
                tuple(self.wires.tolist()),
                str(self.obs.data),
                self.return_type,
            )

        return hash(fingerprint)

    def simplify(self):
        """Reduce the depth of the observable to the minimum.

        Returns:
            .MeasurementProcess: A measurement process with a simplified observable.
        """
        return self if self.obs is None else self.__class__(obs=self.obs.simplify())

    def map_wires(self, wire_map: dict):
        """Returns a copy of the current measurement process with its wires changed according to
        the given wire map.

        Args:
            wire_map (dict): dictionary containing the old wires as keys and the new wires as values

        Returns:
            .MeasurementProcess: new measurement process
        """
        new_measurement = copy.copy(self)
        if self.obs is not None:
            new_measurement.obs = self.obs.map_wires(wire_map=wire_map)
        else:
            new_measurement._wires = Wires([wire_map.get(wire, wire) for wire in self.wires])
        return new_measurement

    def _permute_wires(self, wires: Wires):
        r"""Given an observable which acts on multiple wires, permute the wires to
          be consistent with the device wire order.

          Suppose we are given an observable :math:`\hat{O} = \Identity \otimes \Identity \otimes \hat{Z}`.
          This observable can be represented in many ways:

        .. code-block:: python

              O_1 = qml.Identity(wires=0) @ qml.Identity(wires=1) @ qml.PauliZ(wires=2)
              O_2 = qml.PauliZ(wires=2) @ qml.Identity(wires=0) @ qml.Identity(wires=1)

          Notice that while the explicit tensor product matrix representation of :code:`O_1` and :code:`O_2` is
          different, the underlying operator is identical due to the wire labelling (assuming the labels in
          ascending order are {0,1,2}). If we wish to compute the expectation value of such an observable, we must
          ensure it is identical in both cases. To facilitate this, we permute the wires in our state vector such
          that they are consistent with this swapping of order in the tensor observable.

        .. code-block:: python

              >>> print(O_1.wires)
              <Wires = [0, 1, 2]>
              >>> print(O_2.wires)
              <Wires = [2, 0, 1]>

          We might naively think that we must permute our state vector to match the wire order of our tensor observable.
          We must be careful and realize that the wire order of the terms in the tensor observable DOES NOT match the
          permutation of the terms themselves. As an example we directly compare :code:`O_1` and :code:`O_2`:

          The first term in :code:`O_1` (:code:`qml.Identity(wires=0)`) became the second term in :code:`O_2`.
          By similar comparison we see that each term in the tensor product was shifted one position forward
          (i.e 0 --> 1, 1 --> 2, 2 --> 0). The wires in our permuted quantum state should follow their respective
          terms in the tensor product observable.

          Thus, the correct wire ordering should be :code:`permuted_wires = <Wires = [1, 2, 0]>`. But if we had
          taken the naive approach we would have permuted our state according to
          :code:`permuted_wires = <Wires = [2, 0, 1]>` which is NOT correct.

          This function uses the observable wires and the global device wire ordering in order to determine the
          permutation of the wires in the observable required such that if our quantum state vector is
          permuted accordingly then the amplitudes of the state will match the matrix representation of the observable.

          Args:
              observable (Observable): the observable whose wires are to be permuted.

          Returns:
              permuted_wires (Wires): permuted wires object
        """
        wire_map = dict(zip(wires, range(len(wires))))
        ordered_obs_wire_lst = sorted(self.wires.tolist(), key=lambda label: wire_map[label])
        mapped_wires = [wire_map[w] for w in self.wires]
        permutation = qml.math.argsort(mapped_wires)  # extract permutation via argsort
        return Wires([ordered_obs_wire_lst[index] for index in permutation])


class SampleMeasurement(MeasurementProcess):
    """Sample-based measurement process."""

    @abstractmethod
    def process_samples(
        self,
        samples: Sequence[complex],
        wire_order: Wires,
        shot_range: Tuple[int] = None,
        bin_size: int = None,
    ):
        """Process the given samples.

        Args:
            samples (Sequence[complex]): computational basis samples generated for all wires
            wire_order (Wires): wires determining the subspace that ``samples`` acts on
            shot_range (tuple[int]): 2-tuple of integers specifying the range of samples
                to use. If not specified, all samples are used.
            bin_size (int): Divides the shot range into bins of size ``bin_size``, and
                returns the measurement statistic separately over each bin. If not
                provided, the entire shot range is treated as a single bin.
        """


class StateMeasurement(MeasurementProcess):
    """State-based measurement process."""

    @abstractmethod
    def process_state(self, state: Sequence[complex], wire_order: Wires):
        """Process the given quantum state.

        Args:
            state (Sequence[complex]): quantum state
            wire_order (Wires): wires determining the subspace that ``state`` acts on; a matrix of
                dimension :math:`2^n` acts on a subspace of :math:`n` wires
        """


class CustomMeasurement(MeasurementProcess):
    """Custom measurement process.

    Any class inheriting from this class should define its own ``process`` method, which takes a
    device instance and a tape and returns the result of the measurement process.
    """

    @abstractmethod
    def process(self, tape, device):
        """Process the given tape."""
