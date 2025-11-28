"""
Copyright 2024 Entropica Labs Pte Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

from loom.eka import Eka
from loom.interpreter.applicator import CodeApplicator

from ..code_factory import ColorCode832


# pylint: disable=duplicate-code
class ColorCode832Applicator(CodeApplicator):  # pylint: disable=too-few-public-methods
    """
    Contains the implementation logic for each operation, for the [8,3,2] Color Code.
    """

    def __init__(
        self,
        eka: Eka,
    ):
        # Ensure that all blocks are typed ColorCode832
        if any(not isinstance(block, ColorCode832) for block in eka.blocks):
            raise ValueError("All blocks must be of type ColorCode832.")
        super().__init__(eka)
        # Add the extra operations that are supported by the Color Code 832
        self.supported_operations |= {}

