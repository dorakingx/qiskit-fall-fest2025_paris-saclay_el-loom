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
import numpy as np

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
from loom.eka.utilities.stab_array import StabArray
from loom.eka.utilities.logical_operator_finding import find_logical_operator_set


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

        # Define the 8 physical qubits arranged in a 2D grid: 2 rows x 4 columns
        # All coordinates are 2D tuples (x, y) compatible with SQUARE_2D lattice type.
        # Layout:
        #  (0,0)  (1,0)  (2,0)  (3,0)  (row 0, y=0)
        #  (0,1)  (1,1)  (2,1)  (3,1)  (row 1, y=1)
        # Ancilla qubits are placed in row 2 (y=2) to avoid overlap with data qubits.
        
        # Define stabilizer supports based on cube faces
        # For the [8,3,2] code, we have 5 stabilizers:
        # 3 X-type stabilizers and 2 Z-type stabilizers
        
        # X stabilizers (3 faces of the cube)
        x_stabilizer_supports = [
            # Face 1: qubits (0,0), (1,0), (2,0), (3,0) (top face)
            [(0, 0), (1, 0), (2, 0), (3, 0)],
            # Face 2: qubits (0,0), (1,0), (0,1), (1,1) (front-left face)
            [(0, 0), (1, 0), (0, 1), (1, 1)],
            # Face 3: qubits (2,0), (3,0), (2,1), (3,1) (back-right face)
            [(2, 0), (3, 0), (2, 1), (3, 1)],
        ]
        
        # Z stabilizers (2 faces of the cube)
        z_stabilizer_supports = [
            # Face 4: qubits (0,0), (2,0), (0,1), (2,1) (left face)
            [(0, 0), (2, 0), (0, 1), (2, 1)],
            # Face 5: qubits (1,0), (3,0), (1,1), (3,1) (right face)
            [(1, 0), (3, 0), (1, 1), (3, 1)],
        ]

        # Ancilla qubits placed in row 2 (y=2) to avoid overlap with data qubits
        x_stabilizers = [
            Stabilizer(
                pauli="X" * 4,
                data_qubits=supp,
                ancilla_qubits=[(i, 2)]
            )
            for i, supp in enumerate(x_stabilizer_supports)
        ]
        
        z_stabilizers = [
            Stabilizer(
                pauli="Z" * 4,
                data_qubits=supp,
                ancilla_qubits=[(i + 3, 2)]
            )
            for i, supp in enumerate(z_stabilizer_supports)
        ]

        stabilizers = x_stabilizers + z_stabilizers

        # Automatically derive logical operators from stabilizers using the
        # standard algorithm from Nielsen & Chuang (2011, p.470-471).
        # This ensures the logical operators satisfy canonical commutation relations:
        # - X_L^(i) and Z_L^(j) anti-commute if and only if i == j
        # - All logical operators commute with all stabilizers
        # - X_L operators commute with each other; Z_L operators commute with each other
        
        # Step 1: Collect all data qubits and create coordinate-to-index mapping
        # This mapping is needed to convert between coordinate-based representation
        # (used by Stabilizer/PauliOperator) and index-based representation
        # (used by StabArray/SignedPauliOp)
        all_data_qubits = tuple(
            sorted(set(q for stab in stabilizers for q in stab.data_qubits))
        )
        qubit_to_index = {qubit: i for i, qubit in enumerate(all_data_qubits)}
        index_to_qubit = {i: qubit for qubit, i in qubit_to_index.items()}
        
        # Step 2: Convert stabilizers to SignedPauliOp format and create StabArray
        # The StabArray contains all stabilizer information in a format suitable
        # for the logical operator finding algorithm. For CSS codes like the [8,3,2]
        # color code, the StabArray internally handles the X and Z components separately.
        signed_pauli_ops = [
            stab.as_signed_pauli_op(all_data_qubits) for stab in stabilizers
        ]
        stabarray = StabArray.from_signed_pauli_ops(signed_pauli_ops, validated=False)
        
        # Step 3: Find logical operators automatically using the standard form algorithm
        # This function puts the stabilizer array into standard form and derives
        # the logical X and Z operators that satisfy canonical commutation relations.
        # The algorithm ensures mathematical correctness of the logical operators.
        x_log_stabarray, z_log_stabarray = find_logical_operator_set(stabarray)
        
        # Step 4: Convert logical operators from StabArray back to PauliOperator objects
        # Map the index-based representation back to coordinate-based representation
        # for use in the Block structure.
        logical_x_operators = [
            PauliOperator.from_signed_pauli_op(
                x_log_stabarray[i], index_to_qubit
            )
            for i in range(x_log_stabarray.nstabs)
        ]
        
        logical_z_operators = [
            PauliOperator.from_signed_pauli_op(
                z_log_stabarray[i], index_to_qubit
            )
            for i in range(z_log_stabarray.nstabs)
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

