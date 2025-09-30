"""Functions for wrapping the application custatevec functions to pytket-custatevec specific class instances."""

from collections.abc import Sequence

import numpy as np

from .utils import INSTALL_CUDA_ERROR_MESSAGE

try:
    from cuquantum import ComputeType
    from cuquantum.bindings import custatevec as cusv
    from cuquantum.bindings.custatevec import Pauli as cusvPauli  # type: ignore[import-error]
except ImportError as _cuda_import_err:
    raise RuntimeError(INSTALL_CUDA_ERROR_MESSAGE.format(getattr(_cuda_import_err, "name", None))) from _cuda_import_err


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
    """Apply a quantum gate matrix to a statevector using cuStateVec.

    This function applies a matrix operation to the specified target qubits
    in the given statevector. It supports optional control qubits and their
    corresponding control bit values.

    Args:
        handle (CuStateVecHandle): The cuStateVec handle for managing the
            cuStateVec library context.
        matrix (CuStateVecMatrix): The matrix representing the quantum gate
            to be applied. It should act only on the target qubits.
        statevector (CuStateVector): The statevector to which the matrix
            operation will be applied.
        targets (int | Sequence[int]): The target qubit(s) on which the
            matrix will act. It is assumed this list is provided
            in MSB-to-LSB order, consistent with pytket's convention.
        controls (Sequence[int] | int | None, optional): The control qubit(s)
            for the operation. If None, no control qubits are used. Defaults
            to None.
        control_bit_values (Sequence[int] | int | None, optional): The control
            bit values corresponding to the control qubits. If None, no
            control bit values are used.
            IMPORTANT: 1 means the gate is applied only when the control qubit is in state 1.
        adjoint (bool, optional): Whether to apply the adjoint (conjugate
            transpose) of the matrix. Defaults to False.
        compute_type (ComputeType, optional): The compute precision type to
            use. Defaults to ComputeType.COMPUTE_DEFAULT.
        extra_workspace (int, optional): Additional workspace for the
            operation. Defaults to 0.
        extra_workspace_size_in_bytes (int, optional): Size of the extra
            workspace in bytes. Defaults to 0.

    Returns:
        None: This function modifies the statevector in place.

    Notes:
        - cuStateVec expects the target qubits to be specified in little-endian
          order. This function reverses the order of the targets to comply with
          this requirement.
        - The matrix should only act on the target qubits. cuStateVec internally
          handles embedding the matrix into the full system based on the
          specified target qubits.
        - Since we always set a device memory handler through the CuStateVecHandle,
          the extraWorkspace can be set to null, and the extraWorkspaceSizeInBytes can be set to 0.
    """
    targets = [targets] if isinstance(targets, int) else list(targets)
    # IMPORTANT: After relabling with _qubit_idx_map, cuStateVec.apply_matrix function still
    # requires its list of target indices to be in the LSB-to-MSB order.
    # This reversal adapts our MSB-first list to the LSB-first format cuStateVec requires.
    # Example: For a 4-qubit SWAP(q[2], q[3]), we identify the target qubit indices according to _qubit_idx_map with [1, 0].
    # cuStateVec.apply_matrix requires this to be reversed to [0, 1].
    targets.reverse()  # type: ignore[union-attr]
    controls = [] if controls is None else [controls] if isinstance(controls, int) else list(controls)
    control_bit_values = (
        [] if control_bit_values is None else [control_bit_values] if isinstance(control_bit_values, int) else list(control_bit_values)
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


def pytket_paulis_to_custatevec_paulis(
    pauli_rotation_type: OpType,
    angle_pi: float,
) -> tuple[list[cusvPauli], float]:
    """Map pytket OpType to cuStateVec Pauli and convert angle from multiples of π to radians.

    Args:
        pauli_rotation_type (OpType): The pytket operation type (e.g., Rx, Ry, Rz).
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
    angle_radians = -angle_pi * 0.5 * np.pi
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
    """Apply a Pauli rotation to a statevector using cuStateVec.

    Args:
        handle (CuStateVecHandle): The cuStateVec handle for managing the
            cuStateVec library context.
        paulis (Sequence[cusvPauli]): The sequence of Pauli operators to apply.
        statevector (CuStateVector): The statevector to which the Pauli rotation
            will be applied.
        angle (float): The rotation angle in radians.
        targets (int | Sequence[int]): The target qubit(s) on which the
            Pauli rotation will act.
        controls (Sequence[int] | int | None, optional): The control qubit(s)
            for the operation. If None, no control qubits are used. Defaults
            to None.
        control_bit_values (Sequence[int] | int | None, optional): The control
            bit values corresponding to the control qubits. If None, no
            control bit values are used. Defaults to None.

    Returns:
        None: This function modifies the statevector in place.
    """
    targets = [targets] if isinstance(targets, int) else list(targets)
    # IMPORTANT: After relabling with _qubit_idx_map, cuStateVec.apply_pauli_rotation function still
    # requires its list of target indices to be in the LSB-to-MSB order.
    # This reversal adapts our MSB-first list to the LSB-first format cuStateVec requires.
    # Example: For a 4-qubit SWAP(q[2], q[3]), we identify the target qubit indices according to _qubit_idx_map with [1, 0].
    # cuStateVec.apply_pauli_rotation requires this to be reversed to [0, 1].
    targets.reverse()
    controls = [] if controls is None else [controls] if isinstance(controls, int) else list(controls)
    control_bit_values = (
        [] if control_bit_values is None else [control_bit_values] if isinstance(control_bit_values, int) else list(control_bit_values)
    )

    cusv.apply_pauli_rotation(
        handle=handle.handle,
        sv=statevector.array.data.ptr,
        sv_data_type=statevector.cuda_dtype,
        n_index_bits=statevector.n_qubits,  # TOTAL number of qubits in the statevector
        theta=angle,
        paulis=paulis,
        targets=targets,
        n_targets=len(targets),
        controls=controls,
        control_bit_values=control_bit_values,
        n_controls=len(controls),
    )
