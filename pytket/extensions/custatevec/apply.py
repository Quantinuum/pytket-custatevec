from typing import Sequence, Optional

import numpy as np

from cuquantum import ComputeType
import cuquantum.custatevec as cusv  # type: ignore

from cuquantum.custatevec import Pauli as cusvPauli

from pytket.pauli import Pauli as tkPauli

from .handle import CuStateVecHandle
from .gate_classes import CuStateVecMatrix
from .statevector import CuStateVector



def apply_matrix(
    handle: CuStateVecHandle,
    matrix: CuStateVecMatrix,
    statevector: CuStateVector,
    targets: int | Sequence[int],
    controls: Optional[Sequence[int] | int] = None,
    control_bit_values: Optional[Sequence[int] | int] = None,
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

    cusv.apply_matrix(
        handle=handle.handle,
        sv=statevector.array.data.ptr,
        sv_data_type=statevector.cuda_dtype,
        n_index_bits=matrix.n_qubits,
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


_pytket_pauli_to_custatevec_pauli_map = {
    tkPauli.X: cusvPauli.X,
    tkPauli.Y: cusvPauli.Y,
    tkPauli.Z: cusvPauli.Z,
}


def pytket_paulis_to_custatevec_paulis(pauli: tkPauli) -> cusvPauli:
    return _pytket_pauli_to_custatevec_pauli_map[pauli]


def apply_pauli_rotation(
    handle: CuStateVecHandle,
    paulis: Sequence[cusvPauli],
    statevector: CuStateVector,
    angle: float,
    targets: int | Sequence[int],
    controls: Optional[Sequence[int] | int] = None,
    control_bit_values: Optional[Sequence[int] | int] = None,
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
        handle=handle,
        sv=statevector.array.data.ptr,
        sv_data_type=statevector.cuda_dtype,
        n_index_bits=statevector.n_qubits,
        theta=angle,
        paulis=paulis,
        targets=targets,
        n_targets=len(targets),
        controls=controls,
        control_bit_values=control_bit_values,
        n_controls=len(controls),
    )
