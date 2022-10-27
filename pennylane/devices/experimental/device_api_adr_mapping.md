# Mapping from ADR 052-device_api_spec to required functionality

See [ADR](https://github.com/PennyLaneAI/adrs/blob/master/documents/052-device_api_spec.md) for explicit details and discussions.

The goal of this document is to identify how the proposed API design will address the series of requirements in the above ADR.

1. **It should be possible to extract the configuration from an instantiated device.**
    
    This can easily be achieved through use of a device configuration datastructure/dataclass that holds all standardised operations and supports for each given device. Accessible through an abstract device method:

    ```python
    def get_config(self) -> DeviceConfig:
        return self._device_config
    ```
    Note: `DeviceConfig` requires some consideration for interface and associated data.

2. **Devices can all execute a sequence of tapes, and return a sequence of results.**
    Easily achieved through definition of an `execute` method, which can accept both `tapes`/`QuantumScripts` or lists of both, as:

    ```python
        def execute(self, payload: Union[QuantumScript, List[QuantumScript]]) -> Result:
            ...
    ```
    Note: `Result` also requires some consideration for interface and associated data, as they can be local or remote references.

3. **Devices can register tape transforms and batch transforms to be applied prior to tape execution.**

    This can be handled through explicit addition, or dynamic registration, of preprocessing stages prior to execution as with:

    ```python
        class MyDevice(AbstractDevice):
            "Naive derived class with no provided methods."
            pass

        @TestDeviceBare.register_pre()
        def my_preproc_function(self, qscript: QuantumScript) -> Tuple[Union[QuantumScript, List[QuantumScript]], Callable]:
            "Register a do nothing preprocessing function"
            return tuple(qscript, lambda x: x)
    ```
    Note: This can live inside or outside the device. It is an open question where this will be called --- within `execute` or handled externally, such as by a runtime manager at the `QNode` level.