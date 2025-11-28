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

"""
Verification script for the [8,3,2] Color Code implementation.

This script performs the "Memory Experiment" required by the hackathon:
1. Initialize the code in logical |0⟩⊗3
2. Measure in the initializing Z basis and assert that the result is logical 0
3. Sanity Check 1: Initialize in logical |1⟩⊗3 and verify the measurement is logical 1
4. Sanity Check 2: Initialize in logical |0⟩ and measure in the Logical X basis.
   Verify that the measurement statistics are 50/50 (random) as expected.
"""

import sys
from pathlib import Path

# Add src to path to import loom modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loom.eka import Eka, Lattice
from loom.interpreter import interpret_eka
from loom.eka.operations.code_operation import (
    ResetAllDataQubits,
    MeasureBlockSyndromes,
    MeasureLogicalZ,
    MeasureLogicalX,
)
from loom.executor.eka_circuit_to_stim_converter import EkaCircuitToStimConverter
from loom.eka.utilities.pauli_binary_vector_rep import pauliops_anti_commute, SignedPauliOp
import numpy as np

from loom_color_code_832.code_factory import ColorCode832


def verify_logical_operators(block: ColorCode832):
    """
    Pre-check function to verify that logical operators satisfy the correct
    commutation relations before running simulations.
    
    Checks:
    1. X_L^(i) and Z_L^(j) anti-commute if and only if i == j
    2. All logical operators commute with all stabilizers
    
    Parameters
    ----------
    block : ColorCode832
        The ColorCode832 block to verify
        
    Raises
    ------
    AssertionError
        If any of the commutation checks fail
    """
    print("\n" + "-" * 70)
    print("Pre-check: Verifying Logical Operators")
    print("-" * 70)
    
    # Collect all data qubits for conversion
    all_data_qubits = tuple(sorted(set(q for stab in block.stabilizers for q in stab.data_qubits)))
    
    # Convert logical operators to SignedPauliOp for commutation checks
    x_log_ops = [
        log_x.as_signed_pauli_op(all_data_qubits)
        for log_x in block.logical_x_operators
    ]
    z_log_ops = [
        log_z.as_signed_pauli_op(all_data_qubits)
        for log_z in block.logical_z_operators
    ]
    
    # Convert stabilizers to SignedPauliOp
    stab_ops = [
        stab.as_signed_pauli_op(all_data_qubits)
        for stab in block.stabilizers
    ]
    
    # Check 1: X_L^(i) and Z_L^(j) anti-commute if and only if i == j
    print("Checking logical operator anti-commutation relations...")
    for i, x_op in enumerate(x_log_ops):
        for j, z_op in enumerate(z_log_ops):
            anti_commutes = pauliops_anti_commute(x_op, z_op)
            if i == j:
                assert anti_commutes, (
                    f"Logical X operator {i} and Logical Z operator {j} "
                    f"should anti-commute (i == j), but they commute."
                )
            else:
                assert not anti_commutes, (
                    f"Logical X operator {i} and Logical Z operator {j} "
                    f"should commute (i != j), but they anti-commute."
                )
    print("✓ All logical X and Z operators have correct anti-commutation relations")
    
    # Check 2: All logical operators commute with all stabilizers
    print("Checking that logical operators commute with stabilizers...")
    for i, x_op in enumerate(x_log_ops):
        for j, stab_op in enumerate(stab_ops):
            anti_commutes = pauliops_anti_commute(x_op, stab_op)
            assert not anti_commutes, (
                f"Logical X operator {i} should commute with stabilizer {j}, "
                f"but they anti-commute."
            )
    
    for i, z_op in enumerate(z_log_ops):
        for j, stab_op in enumerate(stab_ops):
            anti_commutes = pauliops_anti_commute(z_op, stab_op)
            assert not anti_commutes, (
                f"Logical Z operator {i} should commute with stabilizer {j}, "
                f"but they anti-commute."
            )
    print("✓ All logical operators commute with all stabilizers")
    
    print("✓ All pre-checks passed! The code structure is mathematically valid.")
    print("-" * 70)


def execute_and_get_logical_result(final_step, logical_qubit_idx=0, seed=42, shots=1):
    """Execute the circuit using Stim and get the logical measurement result."""
    if logical_qubit_idx >= len(final_step.logical_observables):
        return None
    
    # Convert circuit to stim format
    converter = EkaCircuitToStimConverter()
    stim_circuit = converter.convert(final_step)
    
    # Sample the stim circuit
    sampler = stim_circuit.compile_sampler()
    samples = sampler.sample(shots=shots, seed=seed)
    
    # Get the observable index (observables are numbered starting from 1)
    obs_idx = logical_qubit_idx + 1
    if obs_idx > stim_circuit.num_observables:
        return None
    
    # Get the measurement indices for this observable
    obs_instr = stim_circuit[-obs_idx]
    meas_indices = [rec.value for rec in obs_instr.targets_copy()]
    
    # Calculate the parity (XOR) of the measurements
    obs_samples = np.bitwise_xor.reduce(samples[:, meas_indices], axis=1)
    
    # Return the result (for single shot, return the value; for multiple shots, return the first)
    return int(obs_samples[0]) if len(obs_samples) > 0 else None


def run_memory_experiment():
    """Run the memory experiment for the [8,3,2] Color Code."""
    
    print("=" * 70)
    print("Memory Experiment for [8,3,2] Color Code")
    print("=" * 70)
    
    # Create lattice
    lattice = Lattice.square_2d()
    
    # Create the Color Code 832 block
    block = ColorCode832.create(
        lattice=lattice,
        unique_label="color_code_832",
        position=(0, 0),
    )
    
    print(f"\nCreated [8,3,2] Color Code block with label: {block.unique_label}")
    physical_qubits = set(q for s in block.stabilizers for q in s.data_qubits)
    print(f"Number of physical qubits: {len(physical_qubits)}")
    print(f"Number of logical qubits: {len(block.logical_x_operators)}")
    print(f"Number of stabilizers: {len(block.stabilizers)}")
    
    # Pre-check: Verify logical operators before running experiments
    verify_logical_operators(block)
    
    # Test 1: Initialize in logical |0⟩⊗3 and measure in Z basis
    print("\n" + "-" * 70)
    print("Test 1: Initialize in logical |0⟩⊗3 and measure in Z basis")
    print("-" * 70)
    
    operations_1 = [
        ResetAllDataQubits(block.unique_label, state="0"),
        MeasureBlockSyndromes(block.unique_label, n_cycles=1),
        MeasureLogicalZ(block.unique_label, logical_qubit=0),
        MeasureLogicalZ(block.unique_label, logical_qubit=1),
        MeasureLogicalZ(block.unique_label, logical_qubit=2),
    ]
    
    eka_1 = Eka(lattice=lattice, blocks=[block], operations=operations_1)
    final_step_1 = interpret_eka(eka_1, debug_mode=True)
    
    # Get measurement results for all 3 logical qubits
    logical_z_results = []
    for i in range(3):
        result = execute_and_get_logical_result(final_step_1, i)
        logical_z_results.append(result)
    
    print(f"Logical Z measurement results: {logical_z_results}")
    assert all(r == 0 for r in logical_z_results if r is not None), \
        f"Expected all logical Z measurements to be 0, got {logical_z_results}"
    print("✓ Test 1 passed: All logical Z measurements are 0")
    
    # Test 2: Initialize in logical |1⟩⊗3 and measure in Z basis
    print("\n" + "-" * 70)
    print("Test 2: Initialize in logical |1⟩⊗3 and measure in Z basis")
    print("-" * 70)
    
    operations_2 = [
        ResetAllDataQubits(block.unique_label, state="1"),
        MeasureBlockSyndromes(block.unique_label, n_cycles=1),
        MeasureLogicalZ(block.unique_label, logical_qubit=0),
        MeasureLogicalZ(block.unique_label, logical_qubit=1),
        MeasureLogicalZ(block.unique_label, logical_qubit=2),
    ]
    
    eka_2 = Eka(lattice=lattice, blocks=[block], operations=operations_2)
    final_step_2 = interpret_eka(eka_2, debug_mode=True)
    
    # Get measurement results
    logical_z_results_2 = []
    for i in range(3):
        result = execute_and_get_logical_result(final_step_2, i)
        logical_z_results_2.append(result)
    
    print(f"Logical Z measurement results: {logical_z_results_2}")
    assert all(r == 1 for r in logical_z_results_2 if r is not None), \
        f"Expected all logical Z measurements to be 1, got {logical_z_results_2}"
    print("✓ Test 2 passed: All logical Z measurements are 1")
    
    # Test 3: Initialize in logical |0⟩ and measure in Logical X basis (50/50 statistics)
    print("\n" + "-" * 70)
    print("Test 3: Initialize in logical |0⟩ and measure in Logical X basis")
    print("-" * 70)
    
    operations_3 = [
        ResetAllDataQubits(block.unique_label, state="0"),
        MeasureBlockSyndromes(block.unique_label, n_cycles=1),
        MeasureLogicalX(block.unique_label, logical_qubit=0),
    ]
    
    eka_3 = Eka(lattice=lattice, blocks=[block], operations=operations_3)
    final_step_3 = interpret_eka(eka_3, debug_mode=False)
    
    # Run multiple shots to check statistics
    n_shots = 1000
    x_measurement_results = execute_and_get_logical_result(
        final_step_3, 0, seed=42, shots=n_shots
    )
    
    # Get all results by sampling again
    converter = EkaCircuitToStimConverter()
    stim_circuit = converter.convert(final_step_3)
    sampler = stim_circuit.compile_sampler()
    samples = sampler.sample(shots=n_shots, seed=42)
    
    # Get the observable measurement indices
    obs_instr = stim_circuit[-1]  # First observable
    meas_indices = [rec.value for rec in obs_instr.targets_copy()]
    obs_samples = np.bitwise_xor.reduce(samples[:, meas_indices], axis=1)
    
    # Calculate statistics
    num_zeros = np.sum(obs_samples == 0)
    num_ones = np.sum(obs_samples == 1)
    zero_fraction = num_zeros / n_shots
    one_fraction = num_ones / n_shots
    
    print(f"X measurement results: {num_zeros} zeros ({zero_fraction:.2%}), {num_ones} ones ({one_fraction:.2%})")
    
    # Check that we get approximately 50/50 distribution (within 10% tolerance)
    assert 0.4 <= zero_fraction <= 0.6, \
        f"Expected approximately 50/50 distribution, got {zero_fraction:.2%} zeros"
    assert 0.4 <= one_fraction <= 0.6, \
        f"Expected approximately 50/50 distribution, got {one_fraction:.2%} ones"
    
    print("✓ Test 3 passed: Logical X measurements show approximately 50/50 distribution")
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)


if __name__ == "__main__":
    run_memory_experiment()

