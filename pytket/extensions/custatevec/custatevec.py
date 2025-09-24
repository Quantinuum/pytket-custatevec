# Copyright Quantinuum  # noqa: D100
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
from typing import TYPE_CHECKING, Literal

from .utils import INSTALL_CUDA_ERROR_MESSAGE

try:
    import cupy as cp
    import cuquantum.custatevec as cusv
    from cuquantum.bindings._utils import cudaDataType
    from cuquantum.bindings.custatevec import Pauli, StateVectorType
except ImportError as _cuda_import_err:
    raise RuntimeError(INSTALL_CUDA_ERROR_MESSAGE.format(getattr(_cuda_import_err, "name", None))) from _cuda_import_err

import numpy as np
from pytket.circuit import OpType, Qubit
from pytket.utils.operators import QubitPauliOperator

from pytket import pauli

from .apply import (
    apply_matrix,
    apply_pauli_rotation,
    pytket_paulis_to_custatevec_paulis,
)
from .gate_definitions import get_gate_matrix, get_uncontrolled_gate
from .logger import set_logger
from .statevector import CuStateVector
from .utils import _remove_meas_and_implicit_swaps

if TYPE_CHECKING:
    from pytket._tket.circuit import Circuit
    from pytket._tket.unit_id import Bit

    from .handle import CuStateVecHandle

_initial_statevector_dict: dict[str, StateVectorType] = {
    "zero": StateVectorType.ZERO,
    "uniform": StateVectorType.UNIFORM,
    "ghz": StateVectorType.GHZ,
    "w": StateVectorType.W,
}


def _cast_pauli(op: pauli.Pauli) -> Pauli:
    if op == pauli.Pauli.I:
        return Pauli.I
    if op == pauli.Pauli.X:
        return Pauli.X
    if op == pauli.Pauli.Y:
        return Pauli.Y
    if op == pauli.Pauli.Z:
        return Pauli.Z
    raise ValueError(f"Unknown Pauli operator: {op}")


def initial_statevector(
    handle: CuStateVecHandle,
    n_qubits: int,
    sv_type: Literal["zero", "uniform", "ghz", "w"],
    dtype: cudaDataType | None = None,
) -> CuStateVector:
    """Initialize a state vector for a quantum circuit.

    Args:
        handle (CuStateVecHandle): cuStateVec handle for managing state vector operations.
        n_qubits (int): Number of qubits in the state vector.
        sv_type (Literal["zero", "uniform", "ghz", "w"]): Type of initial state vector.
        dtype (cudaDataType | None, optional): Data type for the state vector. Defaults to CUDA_C_64F.

    Returns:
        CuStateVector: The initialized state vector.
    """
    if dtype is None:
        dtype = cudaDataType.CUDA_C_64F
    d = 2**n_qubits
    d_sv = cp.empty(d, dtype=cp.complex128)

    with handle.stream:
        cusv.initialize_state_vector(  # type: ignore  # noqa: PGH003
            handle=handle.handle,
            sv=d_sv.data.ptr,
            sv_data_type=cudaDataType.CUDA_C_64F,
            n_index_bits=n_qubits,
            sv_type=_initial_statevector_dict[sv_type],
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
) -> None:
    """Run a quantum circuit using cuStateVec.

    Args:
        handle (CuStateVecHandle): cuStateVec handle for managing state vector operations.
        circuit (Circuit): The quantum circuit to execute.
        initial_state (CuStateVector | str, optional): Initial state vector or type. Defaults to "zero".
        matrix_dtype (cudaDataType | None, optional): Data type for the operator matrix. Defaults to None, which uses CUDA_C_64F.
        loglevel (int, optional): Logging level. Defaults to logging.WARNING.
        logfile (str | None, optional): Log file path. Defaults to None, which uses console.

    Raises:
        ValueError: If the initial_state is not a valid string or CuStateVector.
        NotImplementedError: If the circuit contains symbolic parameters or unsupported operations.
    """
    if isinstance(initial_state, str):
        if initial_state in {"zero", "uniform", "ghz", "w"}:
            state = initial_statevector(
                handle=handle,
                n_qubits=circuit.n_qubits,
                type=initial_state,  # type: ignore # noqa: PGH003
                dtype=cudaDataType.CUDA_C_64F,
            )
        else:
            raise ValueError(f"Invalid initial_state: {initial_state}")
    else:
        state = initial_state
    if matrix_dtype is None:
        matrix_dtype = cudaDataType.CUDA_C_64F
    _logger = set_logger("RunCircuitLogger", level=loglevel, file=logfile)

    _phase = circuit.phase
    if type(_phase) is float:
        state.apply_phase(_phase)
    else:
        raise NotImplementedError("Symbols not yet supported.")

    # IMPORTANT: _qubit_idx_map matches cuStateVec's little-endian convention
    # (qubit 0 = least significant) with pytket's big-endian (qubit 0 = most significant).
    # Now all operations by the cuStateVec library will act on the correct control and target qubits.
    # Note: Any reordering needs to be done inside run_circuit
    # since get_operator_expectation_value just calls the run_circuit function directly.
    _qubit_idx_map: dict[Qubit, int] = {q: i for i, q in enumerate(sorted(circuit.qubits, reverse=True))}
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
            raise NotImplementedError("Symbolic circuits not yet supported")
        gate_name = op.get_name()
        # Get the relevant, relabeled qubit indices for the operation
        qubits = [_qubit_idx_map[x] for x in com.qubits]

        adjoint = False
        if gate_name[-2:] == "dg":
            adjoint = True
            gate_name = gate_name[:-2]
        uncontrolled_gate, n_controls = get_uncontrolled_gate(gate_name)
        # Since control qubits come before target qubits, we split qubits at n_controls.
        controls, targets = qubits[:n_controls], qubits[n_controls:]

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
            uncontrolled_gate_name_without_parameter = uncontrolled_gate.split("(")[0]
            matrix = get_gate_matrix(
                uncontrolled_gate_name_without_parameter,
                op.params,
                matrix_dtype,
            )
            apply_matrix(
                handle=handle,
                matrix=matrix,
                statevector=state,
                targets=targets,
                controls=controls,
                control_bit_values=[1] * n_controls,  # what value does each of the control qubits need to have to activate the gate
                adjoint=adjoint,  # control_bit_values = [1,...,1] means each gate is applied only when the control qubit is in state 1
            )
    handle.stream.synchronize()


def compute_expectation(
    handle: CuStateVecHandle,
    statevector: CuStateVector,
    operator: QubitPauliOperator,
    circuit: Circuit,
    loglevel: int = logging.WARNING,
    logfile: str | None = None,
) -> np.float64:
    """Compute the expectation value of a QubitPauliOperator on a CuStateVector.

    Args:
        handle (CuStateVecHandle): cuStateVec handle.
        statevector (CuStateVector): The state vector on which to compute the exp. val.
        operator (QubitPauliOperator): The operator for which to compute the exp. val.
        circuit (Circuit): The circuit associated with the state vector.
        loglevel (int, optional): Logging level. Defaults to logging.WARNING.
        logfile (str, optional): Log file path. Defaults to None, which uses console.

    Returns:
        np.complex128: The expectation value of the operator on the state vector.
    """
    if not isinstance(operator, QubitPauliOperator):
        raise TypeError("operator must be a QubitPauliOperator")
    if not isinstance(statevector, CuStateVector):
        raise TypeError("statevector must be a CuStateVector")

    _logger = set_logger("ComputeExpectation", level=loglevel, file=logfile)

    # Map qubits to basis bits
    _qubit_idx_map: dict[Qubit, int] = {q: i for i, q in enumerate(sorted(circuit.qubits, reverse=True))}

    # Collect Pauli terms, basis bits, and coefficients for each string in the operator
    pauli_ops: list[list[Pauli]] = []
    basis_bits: list[list[int]] = []
    coefficients: list[float] = []
    for string, coefficient in operator.map.items():
        coefficients.append(float(coefficient.evalf()))
        operators = list(string.map.items())
        pauli_ops.append([_cast_pauli(op) for _, op in operators] or [Pauli.I])
        basis_bits.append([_qubit_idx_map[q] for q, _ in operators] or [min(_qubit_idx_map.values())])

    # Container for expectation values of each string
    expectation_values = np.empty(len(coefficients), dtype=np.float64)

    with handle.stream:
        cusv.compute_expectations_on_pauli_basis(
            handle=handle.handle,
            sv=statevector.array.data.ptr,
            sv_data_type=statevector.cuda_dtype,
            n_index_bits=statevector.n_qubits,
            expectation_values=expectation_values.ctypes.data,
            pauli_operators_array=pauli_ops,
            n_pauli_operator_arrays=len(pauli_ops),
            basis_bits_array=basis_bits,
            n_basis_bits_array=[len(b) for b in basis_bits],
        )
    handle.stream.synchronize()

    # Compute the weighted sum of expectation values
    expectation_value = coefficients @ expectation_values

    return expectation_value.real
