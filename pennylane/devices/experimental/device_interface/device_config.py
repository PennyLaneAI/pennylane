from dataclasses import dataclass
from .device_helpers import DeviceType

@dataclass
class DeviceConfig:
    """Class for setup, configuration and tracking the supported feature-set and enabled features for a given device"""

    """Configuration dataclass to aid in device setup and initialization.

    Args:
        --- shots (bool): Indicate whether finite-shots are enabled.
        grad (bool): Indicate whether gradients are enabled.
        device_type (DeviceType): Indicate the type of the given device.
    """

    # ---shots: bool = False
    state_access: bool = False
    device_type: DeviceType = DeviceType.UNKNOWN
