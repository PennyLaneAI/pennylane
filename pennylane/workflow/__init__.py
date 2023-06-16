
import pennylane as qml

from pennylane.transforms.core import TransformProgram

class TopLevelObj:


    def __init__(self, func, device, **kwargs):

        self._transform_program = TransformProgram()
        self._untracked_transforms = TransformProgram()

    def __call__(self, *args, **kwargs) -> qml.typing.Result:

    @property
    def transform_program(self) -> TransformProgram:

    
