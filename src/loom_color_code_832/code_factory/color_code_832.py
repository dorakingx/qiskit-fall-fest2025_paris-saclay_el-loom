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

# pylint: disable=duplicate-code
from __future__ import annotations
from functools import cached_property
from uuid import uuid4
from pydantic.dataclasses import dataclass

from loom.eka import (
    Block,
    Lattice,
    LatticeType,
    PauliOperator,
    Stabilizer,
    SyndromeCircuit,
    Circuit,
    Channel,
)
from loom.eka.utilities import dataclass_params


@dataclass(**dataclass_params)
class ColorCode832(Block):
    """
    A sub-class of ``Block`` that represents a [8,3,2] Color Code block.
    The [8,3,2] code encodes 3 logical qubits into 8 physical qubits with distance 2.
    It is based on a cube structure where each vertex represents a physical qubit.
    """

    @classmethod
    def create(  # pylint: disable=too-many-locals
        cls,
        lattice: Lattice,
        unique_label: str = None,
        position: tuple[int, ...] = (0, 0),
        skip_validation: bool = False,
    ) -> ColorCode832:
        """Create [8,3,2] Color Code Block object.
        
        The [8,3,2] Color Code is a quantum error-correcting code that encodes
        3 logical qubits into 8 physical qubits with distance 2. The code is
        based on a cube structure where 8 physical qubits are placed at the
        vertices of a cube. The code has 5 stabilizers (3 X-type and 2 Z-type)
        and 3 pairs of logical operators.

        Parameters
        ----------
        lattice : Lattice
            Lattice on which the block is defined. The qubit indices depend on the type
            of lattice.
        unique_label : str, optional
            Label for the block. It must be unique among all blocks in the initial CRD.
            If no label is provided, a unique label is generated automatically using the
            uuid module.
        position : tuple[int, ...], optional
            Position of the top left corner of the block on the lattice,
            by default (0, 0).
        skip_validation : bool, optional
            Skip validation of the block object, by default False.

        Returns
        -------
        Block
            Block object for the [8,3,2] Color Code.
        """

        # Input validation
        if lattice.lattice_type != LatticeType.SQUARE_2D:
            raise ValueError(
                "The creation of [8,3,2] Color Code blocks is "
                "currently only supported for 2D square lattices. Instead "
                f"the lattice is of type {lattice.lattice_type}."
            )

        if not isinstance(position, tuple) or any(
            not isinstance(x, int) for x in position
        ):
            raise ValueError(
                f"`position` must be a tuple of integers. Got '{position}' instead."
            )

        if unique_label is None:
            unique_label = str(uuid4())

        # Define the 8 physical qubits arranged in a cube structure
        # We'll place them in a 2D grid: 4 qubits in first row, 4 in second row
        # Qubit positions: (x, y, 0) where x,y are grid positions
        # Layout:
        #  0  1  2  3  (first row)
        #  4  5  6  7  (second row)
        
        # Define stabilizer supports based on cube faces
        # For the [8,3,2] code, we have 5 stabilizers:
        # 3 X-type stabilizers and 2 Z-type stabilizers
        
        # X stabilizers (3 faces of the cube)
        x_stabilizer_supports = [
            # Face 1: qubits 0, 1, 2, 3 (top face)
            [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)],
            # Face 2: qubits 0, 1, 4, 5 (front-left face)
            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],
            # Face 3: qubits 2, 3, 6, 7 (back-right face)
            [(2, 0, 0), (3, 0, 0), (2, 1, 0), (3, 1, 0)],
        ]
        
        # Z stabilizers (2 faces of the cube)
        z_stabilizer_supports = [
            # Face 4: qubits 0, 2, 4, 6 (left face)
            [(0, 0, 0), (2, 0, 0), (0, 1, 0), (2, 1, 0)],
            # Face 5: qubits 1, 3, 5, 7 (right face)
            [(1, 0, 0), (3, 0, 0), (1, 1, 0), (3, 1, 0)],
        ]

        x_stabilizers = [
            Stabilizer(
                pauli="X" * 4,
                data_qubits=supp,
                ancilla_qubits=[(i, 0, 1)]
            )
            for i, supp in enumerate(x_stabilizer_supports)
        ]
        
        z_stabilizers = [
            Stabilizer(
                pauli="Z" * 4,
                data_qubits=supp,
                ancilla_qubits=[(i + 3, 0, 1)]
            )
            for i, supp in enumerate(z_stabilizer_supports)
        ]

        stabilizers = x_stabilizers + z_stabilizers

        # Define the logical operators for 3 logical qubits
        # Logical operators must commute with all stabilizers
        
        # Logical X operators (3 logical qubits)
        logical_x_operators = [
            # Logical X1: X on qubits 0, 4
            PauliOperator(
                pauli="XX",
                data_qubits=[(0, 0, 0), (0, 1, 0)],
            ),
            # Logical X2: X on qubits 1, 5
            PauliOperator(
                pauli="XX",
                data_qubits=[(1, 0, 0), (1, 1, 0)],
            ),
            # Logical X3: X on qubits 2, 6
            PauliOperator(
                pauli="XX",
                data_qubits=[(2, 0, 0), (2, 1, 0)],
            ),
        ]
        
        # Logical Z operators (3 logical qubits)
        logical_z_operators = [
            # Logical Z1: Z on qubits 0, 1
            PauliOperator(
                pauli="ZZ",
                data_qubits=[(0, 0, 0), (1, 0, 0)],
            ),
            # Logical Z2: Z on qubits 2, 3
            PauliOperator(
                pauli="ZZ",
                data_qubits=[(2, 0, 0), (3, 0, 0)],
            ),
            # Logical Z3: Z on qubits 4, 5
            PauliOperator(
                pauli="ZZ",
                data_qubits=[(0, 1, 0), (1, 1, 0)],
            ),
        ]

        # Define the syndrome extraction circuits
        x_syndrome_circuits = cls.generate_syndrome_extraction_circuits("XXXX")
        z_syndrome_circuits = cls.generate_syndrome_extraction_circuits("ZZZZ")
        syndrome_circuits = [x_syndrome_circuits, z_syndrome_circuits]

        # Define the stabilizer to circuit mapping
        stabilizer_to_circuit = {
            stab.uuid: x_syndrome_circuits.uuid for stab in x_stabilizers
        } | {stab.uuid: z_syndrome_circuits.uuid for stab in z_stabilizers}

        block = cls(
            unique_label=unique_label,
            stabilizers=stabilizers,
            logical_x_operators=logical_x_operators,
            logical_z_operators=logical_z_operators,
            syndrome_circuits=syndrome_circuits,
            stabilizer_to_circuit=stabilizer_to_circuit,
            skip_validation=skip_validation,
        )

        if position == (0, 0):
            return block

        return block.shift(position)

    @staticmethod
    def generate_syndrome_extraction_circuits(pauli: str) -> SyndromeCircuit:
        """
        Generate syndrome extraction circuit for stabilizers from the [8,3,2] Color Code.
        The prescription followed here is similar to the Steane code: first measure all
        stabilizers of a given type simultaneously. We choose X stabilizers to be
        measured first, thus we need to add idling steps at the end for X stabilizers
        and at the beginning for Z stabilizers. The ancilla is then measured and reset.

        Parameters
        ----------
        pauli: str
            Pauli operator for which the syndrome extraction circuit is generated.

        Returns
        -------
        SyndromeCircuit
            Syndrome extraction circuit for the given Pauli operator.
        """

        # Extract parameters
        name = f"{pauli}_syndrome_extraction"
        pauli_type = pauli[0]

        # Define channels
        data_channels = [Channel(type="quantum", label=f"d{i}") for i in range(4)]
        cbit_channel = Channel(type="classical", label="c0")
        ancilla_channel = Channel(type="quantum", label="a0")

        # Define Hadamard gates
        hadamard1 = tuple([Circuit("H", channels=[ancilla_channel])])
        hadamard2 = tuple([Circuit("H", channels=[ancilla_channel])])

        # Entangling layer
        entangle_ancilla = [
            [Circuit(f"C{p}", channels=[ancilla_channel, data_channels[i]])]
            for i, p in enumerate(pauli)
        ]

        # Add idling step
        if pauli_type == "Z":
            entangle_ancilla = [(), (), (), ()] + entangle_ancilla
        else:
            entangle_ancilla += [(), (), (), ()]

        # Add ancilla measurement and reset
        measurement = tuple(
            [Circuit("Measurement", channels=[ancilla_channel, cbit_channel])]
        )
        reset = tuple([Circuit("Reset_0", channels=[ancilla_channel])])

        # Compose circuit operations as a list
        circuit_list = [reset, hadamard1] + entangle_ancilla + [hadamard2, measurement]

        # Return the syndrome extraction circuit
        return SyndromeCircuit(
            pauli=pauli,
            name=name,
            circuit=Circuit(
                name=name,
                circuit=circuit_list,
                channels=data_channels + [ancilla_channel, cbit_channel],
            ),
        )

    # Instance methods
    def __eq__(self, other: ColorCode832) -> bool:
        if not isinstance(other, ColorCode832):
            raise NotImplementedError(f"Cannot compare ColorCode832 with {type(other)}")
        return super().__eq__(other)

    def shift(
        self, position: tuple[int, ...], new_label: str | None = None
    ) -> ColorCode832:
        return super().shift(position, new_label)

    def rename(self, name: str) -> ColorCode832:
        return super().rename(name)

    @cached_property
    def stabilizers_labels(self) -> dict[str, dict[str, tuple[int, ...]]]:
        return super().stabilizers_labels

