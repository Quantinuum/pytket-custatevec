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

import logging
from typing import Literal

import cupy as cp  # type: ignore
import cuquantum.custatevec as cusv  # type: ignore
from cuquantum.bindings._utils import cudaDataType
from cuquantum.bindings.custatevec import StateVectorType

from pytket.circuit import Bit, Circuit, OpType, Qubit

from .apply import (
    apply_matrix,
    apply_pauli_rotation,
    pytket_paulis_to_custatevec_paulis,
)
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
) -> dict[Qubit, Bit]:
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

    _logger = set_logger("GeneralState", level=loglevel, file=logfile)

    # Remove end-of-circuit measurements and keep track of them separately
    # It also resolves implicit SWAPs
    _measurements: dict[Qubit, Bit]
    circuit, _measurements = _remove_meas_and_implicit_swaps(circuit)

    # Identify each qubit with an index
    # IMPORTANT: Reverse qubit indices to match cuStateVec's little-endian convention
    # (qubit 0 = least significant) vs pytket's big-endian (qubit 0 = most significant).
    _qubit_idx_map: dict[Qubit, int] = {
        q: i for i, q in enumerate(sorted(circuit.qubits, reverse=True))
    }

    # _phase = circuit.phase
    # if type(_phase) is Expr:
    #     raise NotImplementedError("Symbols not yet supported.")
    # state.apply_phase(_phase)
    # Apply the phase from the circuit: sv *= np.exp(1j * np.pi * self._phase)

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
                control_bit_values=[1] * n_controls,
                adjoint=adjoint,
            )

    handle.stream.synchronize()

    return _measurements
