# Copyright Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
##
#     http://www.apache.org/licenses/LICENSE-2.0
##
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import logging
from typing import Literal

import cupy as cp
import cuquantum.custatevec as cusv
import numpy as np
from cuquantum import ComputeType
from cuquantum.bindings._utils import cudaDataType
from cuquantum.bindings.custatevec import StateVectorType

from pytket.circuit import Bit, Circuit, OpType, Qubit
from pytket.extensions.custatevec.gate_classes import CuStateVecMatrix
from pytket.utils.operators import QubitPauliOperator

from .apply import (
    apply_matrix,
    apply_pauli_rotation,
    pytket_paulis_to_custatevec_paulis,
)
from .dtype import cuquantum_to_np_dtype
from .gate_definitions import get_gate_matrix, get_uncontrolled_gate
from .handle import CuStateVecHandle
from .logger import set_logger
from .statevector import CuStateVector
from .utils import _remove_meas_and_implicit_swaps

_initial_statevector_dict: dict[str, StateVectorType] = {
    "zero": StateVectorType.ZERO,
    "uniform": StateVectorType.UNIFORM,
    "ghz": StateVectorType.GHZ,
    "w": StateVectorType.W,
}


def initial_statevector(
    handle: CuStateVecHandle,
    n_qubits: int,
    type: Literal["zero", "uniform", "ghz", "w"],
    dtype: cudaDataType | None = None,
) -> CuStateVector:
    if dtype is None:
        dtype = cudaDataType.CUDA_C_64F
    d = 2**n_qubits
    d_sv = cp.empty(d, dtype=cp.complex128)

    with handle.stream:
        cusv.initialize_state_vector(
            handle.handle,
            d_sv.data.ptr,
            cudaDataType.CUDA_C_64F,
            n_qubits,
            _initial_statevector_dict[type],
        )
    handle.stream.synchronize()
    return CuStateVector(d_sv, dtype)


def run_circuit(
    handle: CuStateVecHandle,
    circuit: Circuit,
    initial_state: CuStateVector | str = "zero",
    matrix_dtype: cudaDataType | None = None,
    loglevel: int = logging.WARNING,
    logfile: str | None = None,
):
    state: CuStateVector
    if type(initial_state) is str:
        state = initial_statevector(
            handle,
            circuit.n_qubits,
            initial_state,
            dtype=cudaDataType.CUDA_C_64F,
        )
    else:
        state = initial_state
    if matrix_dtype is None:
        matrix_dtype = cudaDataType.CUDA_C_64F
    _logger = set_logger("RunCircuitLogger", level=loglevel, file=logfile)

    _phase = circuit.phase
    if type(_phase) is float:
        state.apply_phase(_phase)
    else:
        raise NotImplementedError("Symbols not yet supported.")  # noqa: EM101

    # Identify each qubit with an index
    # IMPORTANT: Reverse qubit indices to match cuStateVec's little-endian convention
    # (qubit 0 = least significant) vs pytket's big-endian (qubit 0 = most significant).
    # Now all operations by the cuStateVec library will be in the correct order.
    # Reordering needs to be done inside the function since get_operator_expectation_value
    # just calls the run_circuit function directly.
    _qubit_idx_map: dict[Qubit, int] = {
        q: i for i, q in enumerate(sorted(circuit.qubits, reverse=True))
    }
    # Remove end-of-circuit measurements and keep track of them separately
    # It also resolves implicit SWAPs
    _measurements: dict[Qubit, Bit]
    circuit, _measurements = _remove_meas_and_implicit_swaps(
        circuit,
    )

    # Apply all gates to the initial state
    commands = circuit.get_commands()
    for com in commands:
        op = com.op
        if len(op.free_symbols()) > 0:
            raise NotImplementedError("Symbolic circuits not yet supported")  # noqa: EM101
        gate_name = op.get_name()
        qubits = [_qubit_idx_map[x] for x in com.qubits]
        uncontrolled_gate, n_controls = get_uncontrolled_gate(gate_name)
        controls, targets = qubits[:n_controls], qubits[n_controls:]

        # TODO: Check if PauliExpBox should go there and what is does
        if op.type in (OpType.Rx, OpType.Ry, OpType.Rz):
            cusv_paulis, angle_radians = pytket_paulis_to_custatevec_paulis(
                pauli_rotation_type=op.type,
                angle_pi=float(op.params[0]),
            )
            apply_pauli_rotation(
                handle=handle,
                paulis=cusv_paulis,
                statevector=state,
                angle=angle_radians,
                targets=targets,
            )
        else:
            adjoint = False
            if gate_name[-2:] == "dg":
                adjoint = True
                gate_name = gate_name[:-2]
            uncontrolled_gate_name_without_parameter = uncontrolled_gate.split("(")[0]
            matrix = get_gate_matrix(
                uncontrolled_gate_name_without_parameter, op.params, matrix_dtype,
            )
            apply_matrix(
                handle=handle,
                matrix=matrix,
                statevector=state,
                targets=targets,
                controls=controls,
                control_bit_values=[1] * n_controls, # 1 means the gate is applied only when the control qubit is in state 1
                adjoint=adjoint,
            )
    handle.stream.synchronize()

    # return _measurements

def compute_expectation(  # noqa: PLR0913
    handle: CuStateVecHandle,
    statevector: CuStateVector,
    operator: QubitPauliOperator,
    circuit: Circuit,
    matrix_dtype: cudaDataType | None = None,
    loglevel: int = logging.WARNING,
    logfile: str | None = None,
) -> np.float64:
    """Compute the expectation value of a QubitPauliOperator on a CuStateVector.

    Args:
        handle (CuStateVecHandle): cuStateVec handle.
        statevector (CuStateVector): The state vector on which to compute the exp. val.
        operator (QubitPauliOperator): The operator for which to compute the exp. val.
        matrix_dtype (cudaDataType, optional): The CUDA data type for operator matrix.
            Defaults to None, which uses CUDA_C_64F.
        loglevel (int, optional): Logging level. Defaults to logging.WARNING.
        logfile (str, optional): Log file path. Defaults to None, which uses console.

    Returns:
        np.complex128: The expectation value of the operator on the state vector.
    """
    if not isinstance(operator, QubitPauliOperator):
        raise TypeError("operator must be a QubitPauliOperator") # noqa: EM101, TRY003
    if not isinstance(statevector, CuStateVector):
        raise TypeError("statevector must be a CuStateVector")  # noqa: EM101, TRY003

    if matrix_dtype is None:
        matrix_dtype = cudaDataType.CUDA_C_64F
    _logger = set_logger("ComputeExpectation", level=loglevel, file=logfile)

    # Convert the operator to a sparse matrix and create a CuStateVecMatrix
    dtype = cuquantum_to_np_dtype(matrix_dtype)
    matrix_array = operator.to_sparse_matrix().toarray()
    matrix = CuStateVecMatrix(
            cp.array(matrix_array, dtype=dtype), matrix_dtype,
        )
    _qubit_idx_map: dict[Qubit, int] = {
        q: i for i, q in enumerate(sorted(circuit.qubits, reverse=True))
    }
    # Match cuStateVec's little-endian convention: Sort basis bits in LSB-to-MSB order
    basis_bits = [_qubit_idx_map[x] for x in operator.all_qubits]

    expectation_value = np.empty(1, dtype=np.complex128)

    with handle.stream:
        cusv.compute_expectation(
            handle=handle.handle,
            sv=statevector.array.data.ptr,
            sv_data_type=statevector.cuda_dtype,
            n_index_bits=statevector.n_qubits,
            expectation_value=expectation_value.ctypes.data, # requires **host** pointer
            expectation_data_type=cudaDataType.CUDA_C_64F,
            matrix=matrix.matrix.data.ptr,
            matrix_data_type=matrix.cuda_dtype,
            layout=cusv.MatrixLayout.COL, # COL -> correct phase for complex exp. val.
            basis_bits=basis_bits,
            n_basis_bits=len(basis_bits),
            compute_type=ComputeType.COMPUTE_DEFAULT,
            extra_workspace=0,
            extra_workspace_size_in_bytes=0,
        )
    handle.stream.synchronize()
    return expectation_value[0]
