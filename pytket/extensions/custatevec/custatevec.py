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
from __future__ import annotations  # type: ignore

from typing import Optional, Literal

import cupy as cp  # type: ignore
import cuquantum.custatevec as cusv  # type: ignore
from cuquantum.bindings.custatevec import StateVectorType

from pytket.circuit import Circuit, PauliExpBox, OpType, Qubit, Bit
from sympy import Expr

from statevector import CuStateVector
from handle import CuStateVecHandle
from logger import set_logger
from utils import _remove_meas_and_implicit_swaps
from dtype import cuquantum_to_np_dtype
from gate_definitions import get_gate_matrix, get_uncontrolled_gate
from apply import apply_matrix, apply_pauli_rotation, pytket_paulis_to_custatevec_paulis

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
    dtype: Optional[cudaDataType] = None,
) -> CuStateVector:
    if dtype is None:
        dtype = cudaDataType.CUDA_C_64F
    d = 2**n_qubits
    d_sv = cp.empty(d, dtype=cuquantum_to_np_dtype(dtype))

    cusv.initialize_state_vector(
        handle.handle,
        d_sv.data.ptr,
        dtype,
        n_qubits,
        _initial_statevector_dict[type],
    )

    return CuStateVector(d_sv, dtype)


def run_circuit(
    handle: CuStateVecHandle,
    circuit: Circuit,
    initial_state: CuStateVector | str = "zero",
    matrix_dtype: Optional[cudaDataType] = None,
    loglevel: int = logging.WARNING,
    logfile: Optional[str] = None,
) -> dict[Qubit, Bit]:
    state : CuStateVector
    if type(initial_state) is str:
        state = initial_statevector(
            handle, initial_state, circuit.n_qubits, dtype=cudaDataType.CUDA_C_64F
        )
    else:
        state = initial_state
    if matrix_dtype is None:
        matrix_dtype = cudaDataType.CUDA_C_64F

    _logger = set_logger("GeneralState", level=loglevel, file=logfile)

    # Remove end-of-circuit measurements and keep track of them separately
    # It also resolves implicit SWAPs
    _measurements : dict[Qubit, Bit]
    circuit, _measurements = _remove_meas_and_implicit_swaps(circuit)

    # Identify each qubit with an index
    _qubit_idx_map : dict[Qubit, int] = {q: i for i, q in enumerate(sorted(circuit.qubits))}

    _phase = circuit.phase
    if type(_phase) is Expr:
        raise NotImplementedError("Symbols not yet supported.")
    state.apply_phase(_phase)

    # Apply all gates to the initial state
    commands = circuit.get_commands()
    for com in commands:
        op = com.op
        if len(op.free_symbols()) > 0:
            raise NotImplementedError("Symbolic circuits not yet supported")
        
        gate_name = op.get_name()
        qubits = [_qubit_idx_map[x] for x in com.qubits]
        uncontrolled_gate, n_controls = get_uncontrolled_gate(gate_name)
        controls, targets = qubits[:n_controls], qubits[n_controls:]

        # TODO: Also check if Rz, Rx, ... should be included in this first branch
        if type(op) is PauliExpBox:
            cusv_paulis = list(map(pytket_paulis_to_custatevec_paulis, op.get_paulis()))
            angle : float = op.get_phase()
            apply_pauli_rotation(
                handle=handle,
                paulis=cusv_paulis,
                statevector=state,
                angle=angle,
                targets=targets
            )
        else:
            adjoint = False
            if gate_name[-2:] == "dg":
                adjoint = True
                gate_name = gate_name[:-2]
            matrix = get_gate_matrix(uncontrolled_gate, op.params, matrix_dtype)
            apply_matrix(
                handle=handle,
                matrix=matrix,
                statevector=state,
                targets=targets,
                controls=controls,
                control_bit_values=[0] * n_controls,
                adjoint=adjoint,
            )
    
    handle.stream.synchronize()

    return _measurements
