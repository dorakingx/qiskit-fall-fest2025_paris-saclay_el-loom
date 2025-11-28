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
Visualization script for the [8,3,2] Color Code implementation.

This script creates a matplotlib visualization showing:
1. Physical qubits as nodes at their (x, y) coordinates
2. X-stabilizers as red polygons
3. Z-stabilizers as blue polygons
4. Logical operators (X_L^(0) and Z_L^(0)) highlighted
5. Qubit annotations with coordinates
"""

import sys
from pathlib import Path

# Add src to path to import loom modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from loom.eka import Lattice
from loom_color_code_832.code_factory import ColorCode832


def order_points_for_polygon(points):
    """
    Order points counterclockwise to form a proper polygon.
    Uses the centroid method to determine ordering.
    
    Parameters
    ----------
    points : list[tuple[float, float]]
        List of (x, y) coordinates
        
    Returns
    -------
    list[tuple[float, float]]
        Points ordered counterclockwise
    """
    if len(points) < 3:
        return points
    
    # Convert to numpy array
    points_array = np.array(points)
    
    # Calculate centroid
    centroid = np.mean(points_array, axis=0)
    
    # Calculate angles from centroid
    angles = []
    for point in points:
        dx = point[0] - centroid[0]
        dy = point[1] - centroid[1]
        angle = np.arctan2(dy, dx)
        angles.append(angle)
    
    # Sort by angle
    sorted_indices = sorted(range(len(points)), key=lambda i: angles[i])
    
    return [points[i] for i in sorted_indices]


def visualize_color_code_832(
    save_path: str | None = None,
    show_ancilla: bool = False,
    figsize: tuple[int, int] = (12, 8),
):
    """
    Visualize the [8,3,2] Color Code structure.
    
    Parameters
    ----------
    save_path : str | None, optional
        Path to save the figure. If None, the figure is displayed but not saved.
    show_ancilla : bool, optional
        Whether to show ancilla qubits. Default is False.
    figsize : tuple[int, int], optional
        Figure size (width, height) in inches. Default is (12, 8).
    """
    # Create the Color Code 832 block
    lattice = Lattice.square_2d()
    block = ColorCode832.create(
        lattice=lattice,
        unique_label="color_code_832",
        position=(0, 0),
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Collect all data qubits
    all_data_qubits = sorted(set(q for stab in block.stabilizers for q in stab.data_qubits))
    
    # Extract qubit coordinates
    qubit_coords = {qubit: qubit for qubit in all_data_qubits}
    
    # Separate X and Z stabilizers
    x_stabilizers = [stab for stab in block.stabilizers if set(stab.pauli) == {"X"}]
    z_stabilizers = [stab for stab in block.stabilizers if set(stab.pauli) == {"Z"}]
    
    # Plot X-stabilizers as red polygons
    for i, stab in enumerate(x_stabilizers):
        qubit_points = [qubit_coords[q] for q in stab.data_qubits]
        ordered_points = order_points_for_polygon(qubit_points)
        polygon = Polygon(
            ordered_points,
            closed=True,
            facecolor="red",
            edgecolor="darkred",
            alpha=0.3,
            linewidth=2,
            label="X-stabilizer" if i == 0 else "",
        )
        ax.add_patch(polygon)
    
    # Plot Z-stabilizers as blue polygons
    for i, stab in enumerate(z_stabilizers):
        qubit_points = [qubit_coords[q] for q in stab.data_qubits]
        ordered_points = order_points_for_polygon(qubit_points)
        polygon = Polygon(
            ordered_points,
            closed=True,
            facecolor="blue",
            edgecolor="darkblue",
            alpha=0.3,
            linewidth=2,
            label="Z-stabilizer" if i == 0 else "",
        )
        ax.add_patch(polygon)
    
    # Plot data qubits
    data_x = [q[0] for q in all_data_qubits]
    data_y = [q[1] for q in all_data_qubits]
    ax.scatter(
        data_x,
        data_y,
        c="black",
        s=200,
        marker="o",
        edgecolors="white",
        linewidths=2,
        zorder=10,
        label="Data Qubit",
    )
    
    # Annotate data qubits with coordinates
    for qubit in all_data_qubits:
        ax.annotate(
            f"({qubit[0]},{qubit[1]})",
            xy=qubit,
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            zorder=11,
        )
    
    # Plot ancilla qubits if requested
    if show_ancilla:
        ancilla_qubits = sorted(
            set(q for stab in block.stabilizers for q in stab.ancilla_qubits)
        )
        ancilla_x = [q[0] for q in ancilla_qubits]
        ancilla_y = [q[1] for q in ancilla_qubits]
        ax.scatter(
            ancilla_x,
            ancilla_y,
            c="gray",
            s=150,
            marker="s",
            edgecolors="white",
            linewidths=2,
            zorder=10,
            label="Ancilla Qubit",
        )
        for qubit in ancilla_qubits:
            ax.annotate(
                f"A({qubit[0]},{qubit[1]})",
                xy=qubit,
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                style="italic",
                zorder=11,
            )
    
    # Overlay Logical Operators (X_L^(0) and Z_L^(0))
    if len(block.logical_x_operators) > 0 and len(block.logical_z_operators) > 0:
        x_log_op = block.logical_x_operators[0]
        z_log_op = block.logical_z_operators[0]
        
        # Plot X_L^(0) - connect all qubits in the operator
        x_log_qubits = x_log_op.data_qubits
        if len(x_log_qubits) > 1:
            # Connect all pairs of qubits to show the full operator support
            x_log_x = [q[0] for q in x_log_qubits]
            x_log_y = [q[1] for q in x_log_qubits]
            
            # Draw lines connecting all qubits (complete graph style)
            for i, q1 in enumerate(x_log_qubits):
                for j, q2 in enumerate(x_log_qubits):
                    if i < j:  # Only draw each edge once
                        ax.plot(
                            [q1[0], q2[0]],
                            [q1[1], q2[1]],
                            color="green",
                            linewidth=3,
                            linestyle="-",
                            alpha=0.5,
                            zorder=5,
                            label="X_L^(0)" if i == 0 and j == 1 else "",
                        )
            
            # Highlight qubits in X_L^(0) with green star markers
            ax.scatter(
                x_log_x,
                x_log_y,
                c="green",
                s=350,
                marker="*",
                edgecolors="darkgreen",
                linewidths=2,
                zorder=12,
                label="X_L^(0) Qubits",
            )
        
        # Plot Z_L^(0) - connect all qubits in the operator
        z_log_qubits = z_log_op.data_qubits
        if len(z_log_qubits) > 1:
            z_log_x = [q[0] for q in z_log_qubits]
            z_log_y = [q[1] for q in z_log_qubits]
            
            # Draw dashed lines connecting all qubits
            for i, q1 in enumerate(z_log_qubits):
                for j, q2 in enumerate(z_log_qubits):
                    if i < j:  # Only draw each edge once
                        ax.plot(
                            [q1[0], q2[0]],
                            [q1[1], q2[1]],
                            color="purple",
                            linewidth=3,
                            linestyle="--",
                            alpha=0.5,
                            zorder=5,
                            label="Z_L^(0)" if i == 0 and j == 1 else "",
                        )
            
            # Highlight qubits in Z_L^(0) with purple diamond markers
            ax.scatter(
                z_log_x,
                z_log_y,
                c="purple",
                s=350,
                marker="D",
                edgecolors="darkviolet",
                linewidths=2,
                zorder=12,
                label="Z_L^(0) Qubits",
            )
    
    # Set plot properties
    ax.set_xlabel("X Coordinate", fontsize=12, fontweight="bold")
    ax.set_ylabel("Y Coordinate", fontsize=12, fontweight="bold")
    ax.set_title("[8,3,2] Color Code Structure", fontsize=16, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_aspect("equal")
    
    # Set axis limits with padding
    all_x = data_x.copy()
    all_y = data_y.copy()
    if show_ancilla:
        ancilla_qubits = sorted(
            set(q for stab in block.stabilizers for q in stab.ancilla_qubits)
        )
        all_x.extend([q[0] for q in ancilla_qubits])
        all_y.extend([q[1] for q in ancilla_qubits])
    
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    padding = 0.5
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    
    # Add legend
    ax.legend(
        loc="upper right",
        fontsize=10,
        framealpha=0.9,
        fancybox=True,
        shadow=True,
    )
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # Create visualization
    visualize_color_code_832(
        save_path="color_code_832_visualization.png",
        show_ancilla=True,
    )

