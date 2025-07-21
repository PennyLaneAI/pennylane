import importlib
from abc import ABC, abstractmethod


class SerializerABC(ABC):
    """
    ABC class that serializes a given entity on construction, and deserializes on call.
    """

    @abstractmethod
    def __init__(self, entity: object, backend_cfg=None):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class SerializerPickle(SerializerABC):
    """
    Serializer class that uses `pickle` for serialization and deserialization.
    """

    def __init__(self, entity: object, *args, **kwargs):
        self._backend = importlib.import_module("pickle")
        self._serialized = self._backend.dumps(entity, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._backend.loads(self._serialized)(*args, **kwargs)


class SerializerDill(SerializerABC):
    """
    Serializer class that uses `dill` for serialization and deserialization.
    """

    def __init__(self, entity: object, *args, **kwargs):
        self._backend = importlib.import_module("dill")
        self._serialized = self._backend.dumps(entity, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._backend.loads(self._serialized)(*args, **kwargs)


class SerializerCloudPickle(SerializerABC):
    """
    Serializer class that uses `cloudpickle` for serialization and deserialization.
    """

    def __init__(self, entity: object, *args, **kwargs):
        self._backend = importlib.import_module("cloudpickle")
        self._serialized = self._backend.dumps(entity, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._backend.loads(self._serialized)(*args, **kwargs)
