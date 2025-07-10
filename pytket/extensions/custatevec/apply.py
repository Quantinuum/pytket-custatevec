from collections.abc import Sequence

import cuquantum.custatevec as cusv  # type: ignore
import numpy as np
from cuquantum import ComputeType
from cuquantum.custatevec import Pauli as cusvPauli

from pytket.circuit import OpType

from .gate_classes import CuStateVecMatrix
from .handle import CuStateVecHandle
from .statevector import CuStateVector


def apply_matrix(
    handle: CuStateVecHandle,
    matrix: CuStateVecMatrix,
    statevector: CuStateVector,
    targets: int | Sequence[int],
    controls: Sequence[int] | int | None = None,
    control_bit_values: Sequence[int] | int | None = None,
    adjoint: bool = False,
    compute_type: ComputeType = ComputeType.COMPUTE_DEFAULT,
    extra_workspace: int = 0,
    extra_workspace_size_in_bytes: int = 0,
) -> None:

    targets = [targets] if targets is int else targets
    if controls is None:
        controls = []
    else:
        controls = [controls] if controls is int else controls
    if control_bit_values is None:
        control_bit_values = []
    else:
        control_bit_values = (
            [control_bit_values] if control_bit_values is int else control_bit_values
        )

    # Note: cuStateVec expects the matrix to act only on the target qubits.
    # For example, even in a multi-qubit system (e.g., 2 qubits),
    # applying a single-qubit gate like X only requires a 2x2 matrix.
    # cuStateVec internally handles embedding it into the full system
    # based on the specified target qubit(s).
    cusv.apply_matrix(
        handle=handle.handle,
        sv=statevector.array.data.ptr,
        sv_data_type=statevector.cuda_dtype,
        n_index_bits=statevector.n_qubits,  # TOTAL number of qubits in the statevector
        matrix=matrix.matrix.data.ptr,
        matrix_data_type=matrix.cuda_dtype,
        layout=cusv.MatrixLayout.ROW,
        adjoint=adjoint,
        targets=targets,
        n_targets=len(targets),
        controls=controls,
        control_bit_values=control_bit_values,
        n_controls=len(controls),
        compute_type=compute_type,
        extra_workspace=extra_workspace,
        extra_workspace_size_in_bytes=extra_workspace_size_in_bytes,
    )


def pytket_paulis_to_custatevec_paulis(pauli_rotation_type: OpType, angle_pi: float) -> tuple[list[cusvPauli], float]:
    """Map pytket OpType to cuStateVec Pauli and convert angle from multiples of π to radians.

    Args:
        op_type (OpType): The pytket operation type (e.g., Rx, Ry, Rz).
        angle_pi (float): The angle in multiples of π.

    Returns:
        tuple[list[cusvPauli], float]: A list of cuStateVec Pauli(s) and the angle in radians.
    """
    _pytket_pauli_to_custatevec_pauli_map = {
        OpType.Rx: [cusvPauli.X],
        OpType.Ry: [cusvPauli.Y],
        OpType.Rz: [cusvPauli.Z],
    }
    if pauli_rotation_type not in _pytket_pauli_to_custatevec_pauli_map:
        raise ValueError(f"Unsupported OpType: {pauli_rotation_type}")

    paulis = _pytket_pauli_to_custatevec_pauli_map[pauli_rotation_type]
    # cuStateVec's apply_pauli_rotation applies exp(i*angle_radians*Pauli),
    # where angle_radians is in radians. The input angle from pytket
    # is in multiples of π, so we convert it to radians. Additionally,
    # we apply a factor of 0.5 with a negative sign to render the
    # Pauli rotation an actual rotation gate in the conventional definition.
    angle_radians = - angle_pi * 0.5 * np.pi
    return paulis, angle_radians


def apply_pauli_rotation(
    handle: CuStateVecHandle,
    paulis: Sequence[cusvPauli],
    statevector: CuStateVector,
    angle: float,
    targets: int | Sequence[int],
    controls: Sequence[int] | int | None = None,
    control_bit_values: Sequence[int] | int | None = None,
) -> None:
    targets = [targets] if targets is int else targets
    if controls is None:
        controls = []
    else:
        controls = [controls] if controls is int else controls
    if control_bit_values is None:
        control_bit_values = []
    else:
        control_bit_values = (
            [control_bit_values] if control_bit_values is int else control_bit_values
        )

    cusv.apply_pauli_rotation(
        handle=handle.handle,
        sv=statevector.array.data.ptr,
        sv_data_type=statevector.cuda_dtype,
        n_index_bits=statevector.n_qubits, # TOTAL number of qubits in the statevector
        theta=angle,
        paulis=paulis,
        targets=targets,
        n_targets=len(targets),
        controls=controls,
        control_bit_values=control_bit_values,
        n_controls=len(controls),
    )
