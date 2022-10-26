from enum import IntEnum, auto
from abc import ABC


class DeviceType(IntEnum):
    "Easily distinguish between physical and virtual devices"
    VIRTUAL = auto()
    PHYSICAL = auto()
    UNKNOWN = auto()


class RegistrationsMetaclass(type, ABC):
    def __new__(cls, name, bases, name_space):
        if not bases:
            return type.__new__(cls, name, bases, name_space)
        return super().__new__(cls, name, bases, dict(name_space, registrations={}))
