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
from typing import TYPE_CHECKING
from uuid import uuid4

import cupy as cp
import cuquantum.custatevec as cusv
import numpy as np
from cuquantum import cudaDataType
from cuquantum.bindings.custatevec import SamplerOutput

from pytket._tket.circuit import Circuit
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
from pytket.extensions.custatevec.gate_definitions import _control_to_gate_map, gate_list
from pytket.extensions.custatevec.handle import CuStateVecHandle
from pytket.passes import (
    BasePass,
    CustomPass,
    DecomposeBoxes,
    FullPeepholeOptimise,
    RemoveRedundancies,
    SequencePass,
    SynthesiseTket,
)
from pytket.predicates import (
    NoBarriersPredicate,
    NoClassicalControlPredicate,
    NoMidMeasurePredicate,
    NoSymbolsPredicate,
    Predicate,
)
from pytket.utils.operators import QubitPauliOperator
from pytket.utils.outcomearray import OutcomeArray
from pytket.utils.results import KwargTypes

from .._metadata import __extension_name__, __extension_version__  # noqa: TID252

if TYPE_CHECKING:
    from pytket.circuit import Qubit


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
        return preds  # noqa: RET504

    def rebase_pass(self) -> BasePass:
        """This method returns a dummy pass that does nothing, since there is no need to rebase.

        It is provided by requirement of a child of Backend,
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
        assert optimisation_level in range(3)  # noqa: S101
        seq = [
            DecomposeBoxes(),
            RemoveRedundancies(),
        ]  # Decompose boxes into basic gates

        # NOTE: these are the standard passes used in TKET backends. I haven't
        # benchmarked what's their effect on the simulation time.
        if optimisation_level == 1:
            seq.append(SynthesiseTket())  # Optional fast optimisation
        elif optimisation_level == 2:  # noqa: PLR2004
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
    def process_circuits(  # noqa: D417
        self,
        circuits: Circuit | Sequence[Circuit],
        n_shots: int,
        seed: int,
        valid_check: bool,
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
            # All currently implemented gates including controlled gates
            gate_set={gate.name for gate in gate_list}.union(_control_to_gate_map.keys()),  # type: ignore[no-untyped-call]
            misc={"characterisation": None},
        )

    def process_circuits(  # noqa: D417
        self,
        circuits: Circuit | Sequence[Circuit],
        n_shots: int = 0,  # noqa: ARG002
        seed: int = 0,  # noqa: ARG002
        valid_check: bool = True,
        **kwargs: KwargTypes,  # noqa: ARG002
    ) -> list[ResultHandle]:
        """Submits circuits to the backend for running.

        The results will be stored in the backend's result cache to be retrieved by the
        corresponding get_<data> method.

        Args:
            circuits: List of circuits to be submitted.
            n_shots: Number of shots in case of shot-based calculation.
                This is unused, since this backend does not support shots.
            seed: Seed for random number generation.
                This is unused, since this backend does not support shots.
            valid_check: Whether to check for circuit correctness.

        Returns:
            Results handle objects.
        """
        # Ensure circuits is always a sequence
        circuits = [circuits] if isinstance(circuits, Circuit) else circuits

        if valid_check:
            self._check_all_circuits(circuits, nomeasure_warn=False)

        handle_list = []
        for circuit in circuits:
            with CuStateVecHandle() as libhandle:
                sv = initial_statevector(
                    handle=libhandle,
                    n_qubits=circuit.n_qubits,
                    sv_type="zero",
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
        circuit: Circuit,
        operator: QubitPauliOperator,
    ) -> np.float64:
        """Calculate the expectation value of a QubitPauliOperator given a quantum state prepared by a circuit.

        This method computes the expectation value of a specified operator with respect
        to the quantum state generated by the provided state preparation circuit. It
        leverages cuStateVec for efficient statevector simulation and expectation value
        computation.

        Args:
            circuit (Circuit): The quantum circuit that prepares the desired
                quantum state.
            operator (QubitPauliOperator): The operator for which the expectation value
                is to be calculated.

        Returns:
            np.float64: The computed expectation value of the operator with respect to
            the quantum state.
        """
        with CuStateVecHandle() as libhandle:
            sv = initial_statevector(
                handle=libhandle,
                n_qubits=circuit.n_qubits,
                sv_type="zero",
                dtype=cudaDataType.CUDA_C_64F,
            )
            run_circuit(libhandle, circuit, sv)
            return compute_expectation(libhandle, sv, operator, circuit)


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
            # All currently implemented gates including controlled gates
            gate_set={gate.name for gate in gate_list}.union(_control_to_gate_map.keys()),  # type: ignore[no-untyped-call]
            misc={"characterisation": None},
        )

    def process_circuits(  # noqa: D417
        self,
        circuits: Circuit | Sequence[Circuit],
        n_shots: int,
        seed: int = 4,  # type: ignore[override]
        valid_check: bool = True,
        **kwargs: KwargTypes,  # noqa: ARG002
    ) -> list[ResultHandle]:
        """Submits circuits to the backend for running and returns result handles.

        Args:
            circuits: List of circuits to be submitted.
            n_shots: Number of shots for shot-based calculation.
            seed: Seed for random number generation.
            valid_check: Whether to check for circuit correctness.

        Returns:
            List of result handles for the submitted circuits.
        """
        # Ensure circuits is always a sequence
        circuits = [circuits] if isinstance(circuits, Circuit) else circuits

        if valid_check:
            self._check_all_circuits(circuits, nomeasure_warn=False)

        handle_list = []
        for circuit in circuits:
            with CuStateVecHandle() as libhandle:
                sv = initial_statevector(
                    handle=libhandle,
                    n_qubits=circuit.n_qubits,
                    sv_type="zero",
                    dtype=cudaDataType.CUDA_C_64F,
                )
                run_circuit(libhandle, circuit, sv)

                # IMPORTANT: _qubit_idx_map matches cuStateVec's little-endian convention
                # (qubit 0 = least significant) with pytket's big-endian (qubit 0 = most significant).
                # Now all operations by the cuStateVec library will be in the correct order.
                _qubit_idx_map: dict[Qubit, int] = {q: i for i, q in enumerate(sorted(circuit.qubits, reverse=True))}
                # Get relabeled qubit indices that will be measured
                measured_qubits = [_qubit_idx_map[x] for x in circuit.qubit_readout]
                # IMPORTANT: After relabling with _qubit_idx_map, cuStateVec.sampler_sample function still
                # requires its list of measured qubits to be in the LSB-to-MSB order.
                # This reversal adapts our MSB-first list to the LSB-first format cuStateVec requires.
                measured_qubits.reverse()

                sampler_descriptor, size_t = cusv.sampler_create(  # type: ignore[no-untyped-call]
                    handle=libhandle.handle,
                    sv=sv.array.data.ptr,
                    sv_data_type=cudaDataType.CUDA_C_64F,
                    n_index_bits=sv.n_qubits,
                    n_max_shots=n_shots,
                )

                bit_strings_int64 = np.empty((n_shots, 1), dtype=np.int64)  # needs to be int64

                # Generate random numbers for sampling
                rng = np.random.default_rng(seed)
                randnums = rng.random(n_shots, dtype=np.float64).tolist()

                cusv.sampler_preprocess(  # type: ignore[no-untyped-call]
                    handle=libhandle.handle,
                    sampler=sampler_descriptor,
                    extra_workspace=0,
                    extra_workspace_size_in_bytes=0,
                )

                cusv.sampler_sample(  # type: ignore[no-untyped-call]
                    handle=libhandle.handle,
                    sampler=sampler_descriptor,
                    bit_strings=bit_strings_int64.ctypes.data,
                    bit_ordering=measured_qubits,
                    bit_string_len=len(measured_qubits),
                    randnums=randnums,
                    n_shots=n_shots,
                    output=SamplerOutput.RANDNUM_ORDER,
                )

                cusv.sampler_destroy(sampler_descriptor)  # type: ignore[no-untyped-call]

            handle = ResultHandle(str(uuid4()))

            # Reformat bit_strings from list of 64-bit signed integer (memory-efficient
            # way for custatevec to save many shots) to list of binaries for OutcomeArray
            bit_strings_binary = [format(s, f"0{len(measured_qubits)}b") for s in bit_strings_int64.flatten().tolist()]
            bit_strings_binary = [tuple(map(int, binary)) for binary in bit_strings_binary]  # type: ignore[misc]

            # In order to be able to use the BackendResult functionality,
            # we only pass the array of the statevector to BackendResult
            self._cache[handle] = {
                "result": BackendResult(
                    state=cp.asnumpy(sv.array),
                    shots=OutcomeArray.from_readouts(bit_strings_binary),
                ),
            }
            handle_list.append(handle)
        return handle_list
