# Copyright Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Methods to allow tket circuits to be run on the cuStateVec simulator."""

from abc import abstractmethod
from collections.abc import Sequence
from typing import List
from uuid import uuid4

import cupy as cp
import cuquantum.custatevec as cusv  # type: ignore
import numpy as np
from cuquantum import cudaDataType

from pytket._tket.circuit import Circuit, OpType
from pytket.backends.backend import Backend
from pytket.backends.backend_exceptions import CircuitNotRunError
from pytket.backends.backendinfo import BackendInfo
from pytket.backends.backendresult import BackendResult
from pytket.backends.resulthandle import ResultHandle, _ResultIdTuple
from pytket.backends.status import CircuitStatus, StatusEnum
from pytket.extensions.custatevec.custatevec import (
    compute_expectation,
    initial_statevector,
    run_circuit,
)
from pytket.extensions.custatevec.handle import CuStateVecHandle
from pytket.passes import (  # type: ignore
    BasePass,
    CustomPass,  # type: ignore
    DecomposeBoxes,  # type: ignore
    FullPeepholeOptimise,  # type: ignore
    RemoveRedundancies,
    SequencePass,
    SynthesiseTket,
)
from pytket.predicates import (  # type: ignore
    NoBarriersPredicate,
    NoClassicalControlPredicate,
    NoMidMeasurePredicate,
    NoSymbolsPredicate,
    Predicate,
)
from pytket.utils.outcomearray import OutcomeArray
from pytket.utils.results import KwargTypes

from .._metadata import __extension_name__, __extension_version__
from ..statevector import CuStateVector
from ..utils import _remove_meas_and_implicit_swaps


class _CuStateVecBaseBackend(Backend):
    """A pytket Backend wrapping around the ``GeneralState`` simulator."""

    _persistent_handles = False

    def __init__(self) -> None:
        """Constructs a new cuStateVec backend object."""
        super().__init__()

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return (str,)

    @property
    def required_predicates(self) -> list[Predicate]:
        """Returns the minimum set of predicates that a circuit must satisfy.

        Predicates need to be satisfied before the circuit can be successfully run on
        this backend.

        Returns:
            List of required predicates.
        """
        preds = [
            NoSymbolsPredicate(),
            NoClassicalControlPredicate(),
            NoMidMeasurePredicate(),
            NoBarriersPredicate(),
        ]
        return preds

    def rebase_pass(self) -> BasePass:
        """This method returns a dummy pass that does nothing, since there is
        no need to rebase. It is provided by requirement of a child of Backend,
        but it should not be included in the documentation.
        """
        return CustomPass(lambda circ: circ)  # Do nothing

    def default_compilation_pass(self, optimisation_level: int = 0) -> BasePass:
        """Returns a default compilation pass.

        A suggested compilation pass that will guarantee the resulting circuit
        will be suitable to run on this backend with as few preconditions as
        possible.

        Args:
            optimisation_level: The level of optimisation to perform during
                compilation. Level 0 just solves the device constraints without
                optimising. Level 1 additionally performs some light optimisations.
                Level 2 adds more intensive optimisations that can increase compilation
                time for large circuits. Defaults to 0.

        Returns:
            Compilation pass guaranteeing required predicates.
        """
        assert optimisation_level in range(3)
        seq = [
            DecomposeBoxes(),
            RemoveRedundancies(),
        ]  # Decompose boxes into basic gates

        # NOTE: these are the standard passes used in TKET backends. I haven't
        # benchmarked what's their effect on the simulation time.
        if optimisation_level == 1:
            seq.append(SynthesiseTket())  # Optional fast optimisation
        elif optimisation_level == 2:
            seq.append(FullPeepholeOptimise())  # Optional heavy optimisation
        seq.append(self.rebase_pass())  # Map to target gate set
        return SequencePass(seq)

    def circuit_status(self, handle: ResultHandle) -> CircuitStatus:
        """Returns circuit status object.

        Returns:
            CircuitStatus object.

        Raises:
            CircuitNotRunError: if there is no handle object in cache.
        """
        if handle in self._cache:
            return CircuitStatus(StatusEnum.COMPLETED)
        raise CircuitNotRunError(handle)

    @abstractmethod
    def process_circuits(
        self,
        circuits: Sequence[Circuit],
        n_shots: int | Sequence[int] | None = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> list[ResultHandle]:
        """Submits circuits to the backend for running.

        The results will be stored in the backend's result cache to be retrieved by the
        corresponding get_<data> method.

        Args:
            circuits: List of circuits to be submitted.
            n_shots: Number of shots in case of shot-based calculation.
            valid_check: Whether to check for circuit correctness.

        Returns:
            Results handle objects.

        Raises:
            TypeError: If global phase is dependent on a symbolic parameter.
        """
        ...


class CuStateVecStateBackend(_CuStateVecBaseBackend):
    """A pytket Backend using ``GeneralState`` to obtain state vectors."""

    _supports_state = True
    _supports_expectation = True
    _expectation_allows_nonhermitian = True

    def __init__(self) -> None:
        """Constructs a new cuStateVec backend object."""
        super().__init__()

    @property
    def backend_info(self) -> BackendInfo | None:
        """Returns information on the backend."""
        return BackendInfo(
            name="CuStateVecStateBackend",
            architecture=None,
            device_name="NVIDIA GPU",
            version=__extension_name__ + "==" + __extension_version__,
            # The only constraint to the gateset is that they must be unitary matrices
            # or end-of-circuit measurements. These constraints are already specified
            # in the required_predicates of the backend. The empty set for gateset is
            # meant to be interpreted as "all gates".
            # TODO: list all gates in a programmatic way?
            gate_set=set(),
            misc={"characterisation": None},
        )

    def process_circuits(
        self,
        circuits: Sequence[Circuit],
        n_shots: int | Sequence[int] | None = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> list[ResultHandle]:
        """Submits circuits to the backend for running.

        The results will be stored in the backend's result cache to be retrieved by the
        corresponding get_<data> method.

        Args:
            circuits: List of circuits to be submitted.
            n_shots: Number of shots in case of shot-based calculation.
                This should be ``None``, since this backend does not support shots.
            valid_check: Whether to check for circuit correctness.

        Returns:
            Results handle objects.
        """
        # TODO Valid check and compilation pass
        handle_list = []
        for circuit in circuits:
            with CuStateVecHandle() as libhandle:
                sv = initial_statevector(
                    libhandle,
                    circuit.n_qubits,
                    "zero",
                    dtype=cudaDataType.CUDA_C_64F,
                )
                run_circuit(libhandle, circuit, sv)
            handle = ResultHandle(str(uuid4()))
            # In order to be able to use the BackendResult functionality,
            # we only pass the array of the statevector to BackendResult
            self._cache[handle] = {"result": BackendResult(state=cp.asnumpy(sv.array))}
            handle_list.append(handle)
        return handle_list

    def get_operator_expectation_value(
        self,
        state_circuit,
        operator,
    ):
        with CuStateVecHandle() as libhandle:
            sv = initial_statevector(
                    libhandle,
                    state_circuit.n_qubits,
                    "zero",
                    dtype=cudaDataType.CUDA_C_64F,
                )
            run_circuit(libhandle, state_circuit, sv)
            operator_expectation_value = compute_expectation(libhandle, sv, operator)

        return operator_expectation_value

class CuStateVecShotsBackend(_CuStateVecBaseBackend):
    """A pytket Backend using ``GeneralState`` to obtain shots."""

    _supports_shots = True
    _supports_counts = True

    def __init__(self) -> None:
        """Constructs a new cuStateVec backend object."""
        super().__init__()

    @property
    def backend_info(self) -> BackendInfo | None:
        """Returns information on the backend."""
        return BackendInfo(
            name="CuStateVecShotsBackend",
            architecture=None,
            device_name="NVIDIA GPU",
            version=__extension_name__ + "==" + __extension_version__,
            # The only constraint to the gateset is that they must be unitary matrices
            # or end-of-circuit measurements. These constraints are already specified
            # in the required_predicates of the backend. The empty set for gateset is
            # meant to be interpreted as "all gates".
            # TODO: list all gates in a programmatic way?
            gate_set=set(),
            misc={"characterisation": None},
        )

    def process_circuits(
    self,
    circuits: Sequence[Circuit],
    n_shots: int | Sequence[int] | None = None,
    valid_check: bool = True,
    **kwargs: KwargTypes,
) -> List[ResultHandle]:
        """Submits circuits to the backend for running and stores sampled shots & reordered statevector."""
        handle_list: List[ResultHandle] = []

        # Normalize n_shots into a list for each circuit
        if isinstance(n_shots, int) or n_shots is None:
            n_shots_list = [n_shots] * len(circuits)
        else:
            n_shots_list = list(n_shots)

        for circ_idx, circuit in enumerate(circuits):
            # Save original classical bit list BEFORE removing measurements
            original_bits = list(circuit.bits)
            shots_for_this = n_shots_list[circ_idx] or 1

            with CuStateVecHandle() as libhandle:
                # 1. Allocate |0...0> statevector on GPU
                sv = initial_statevector(
                    libhandle,
                    circuit.n_qubits,
                    "zero",
                    dtype=cudaDataType.CUDA_C_64F,
                )

                # 2. Remove end-of-circuit measurements
                circuit_no_meas, measurements = _remove_meas_and_implicit_swaps(circuit)

                # 3. Simulate stripped circuit
                run_circuit(libhandle, circuit_no_meas, sv)

                # 4. Which qubits were measured?
                measured_qubits = list(measurements.keys())

                # 5. Sample bitstrings for measured qubits using correct backend→pytket mapping
                qubit_shots = self.sample_circuit(
                    circ=circuit,             # ORIGINAL circuit for qubit ordering
                    state=sv,
                    qubits=measured_qubits,   # only measured qubits
                    n_shots=shots_for_this,
                )  # shape (shots_for_this, len(measured_qubits))

                # 6. Pad full shot table with zeros for all classical bits
                all_shots = np.zeros((shots_for_this, len(original_bits)), dtype=int)

                # 7. Fill each measured qubit’s sampled bits into the correct classical bit column
                for i, q in enumerate(measured_qubits):
                    bit = measurements[q]              # classical Bit for this qubit
                    col_idx = original_bits.index(bit) # its column position
                    all_shots[:, col_idx] = qubit_shots[:, i]

                # 8. Fix statevector ordering for pytket (LSB → MSB)
                raw_sv = cp.asnumpy(sv.array)
                fixed_sv = _reorder_state_for_pytket(raw_sv, circuit.n_qubits)

                # 9. Cache result
                res_bits   = original_bits
                res_qubits = list(circuit.qubits)
                handle     = ResultHandle(str(uuid4()))

                self._cache[handle] = {
                    "result": BackendResult(
                        c_bits=res_bits,
                        shots=OutcomeArray.from_readouts(all_shots),
                        state=fixed_sv
                    ),
                }
                handle_list.append(handle)

        return handle_list

    def sample_circuit(
        self,
        circ: Circuit,
        state: CuStateVector,
        qubits: Sequence[int],
        n_shots: int = 1024,
        seed: int | None = None,
    ) -> np.ndarray:
        """Sample bitstrings from the given statevector using inverse transform sampling.

        Args:
            circ: pytket Circuit to determine qubit order.
            state: cuStateVec statevector.
            qubits: Only these qubits will be included in output shots.
            n_shots: Number of samples.
            seed: Optional RNG seed for reproducibility.

        Returns:
            NumPy array of shape (n_shots, len(qubits)) with sampled bits in pytket logical order.
        """
        n_qubits = len(circ.qubits)

        # Optional deterministic seeding
        if seed is not None:
            cp.random.seed(seed)

        # 1. Compute probabilities and CDF
        probs = cp.abs(state.array) ** 2
        cdf = cp.cumsum(probs)

        # 2. Draw random points in [0,1)
        points = cp.random.rand(n_shots)

        # 3. Sample computational basis indices (inverse transform sampling)
        indices = cp.searchsorted(cdf, points)  # shape (n_shots,)

        # 4. Convert indices → backend bitstrings (LSB-first: bit0=q0)
        bit_shifts = cp.arange(n_qubits, dtype=indices.dtype)
        backend_bitmasks = (indices[:, None] >> bit_shifts[None, :]) & 1  # shape (shots, n_qubits)

        # 5. Map backend bit positions -> pytket logical qubits
        # cuStateVec LSB-first -> pytket MSB-first: reversed qubit order
        backend_order = list(reversed(circ.qubits))  # MSB-first
        backend_indices = [backend_order.index(q) for q in qubits]

        # 6. Pick only measured qubits in pytket logical order
        filtered_shots = backend_bitmasks[:, backend_indices]

        # 7. Return as NumPy array
        return cp.asnumpy(filtered_shots)



    def get_qubits_and_bits(circuit):
        measure_map = dict()
        measured_units = (
            set()
        )  # Track measured Qubits/used Bits to identify mid-circuit measurement

        for command in circuit:
            optype = command.op.type
            if optype == OpType.Measure:
                measure_map[command.args[0]] = command.args[1]
                measured_units.add(command.args[0])
                measured_units.add(command.args[1])

        return measure_map


def _check_all_unitary_or_measurements(circuit: Circuit) -> bool:
    """Auxiliary function for custom predicate"""
    try:
        for cmd in circuit:
            if cmd.op.type != OpType.Measure:
                cmd.op.get_unitary()
        return True
    except:
        return False

def _reorder_state_for_pytket(state: np.ndarray, n_qubits: int) -> np.ndarray:
    """Convert cuStateVec LSB-first statevector to pytket's MSB-first convention."""
    reshaped = state.reshape([2] * n_qubits)
    # reverse axes: [n-1, ..., 1, 0]
    reordered = np.transpose(reshaped, axes=list(range(n_qubits - 1, -1, -1)))
    return reordered.flatten()
