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
from uuid import uuid4

import cupy as cp
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
from pytket.utils.operators import QubitPauliOperator
from pytket.utils.results import KwargTypes

from .._metadata import __extension_name__, __extension_version__


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
        state_circuit: Circuit,
        operator: QubitPauliOperator,
    ) -> np.float64:
        """Calculate the expectation value of a QubitPauliOperator given a quantum state prepared by a circuit.

        This method computes the expectation value of a specified operator with respect
        to the quantum state generated by the provided state preparation circuit. It
        leverages cuStateVec for efficient statevector simulation and expectation value
        computation.

        Args:
            state_circuit (Circuit): The quantum circuit that prepares the desired
                quantum state.
            operator (QubitPauliOperator): The operator for which the expectation value
                is to be calculated.

        Returns:
            np.float64: The computed expectation value of the operator with respect to
            the quantum state.
        """  # noqa: E501
        with CuStateVecHandle() as libhandle:
            sv = initial_statevector(
                libhandle,
                state_circuit.n_qubits,
                "zero",
                dtype=cudaDataType.CUDA_C_64F,
            )
            run_circuit(libhandle, state_circuit, sv)
            return compute_expectation(libhandle, sv, operator)


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
    n_shots: int | Sequence[int],
    seed: int | None,
    valid_check: bool = True,
    **kwargs: KwargTypes,
) -> list[ResultHandle]:

        import cuquantum.custatevec as cusv
        from cuquantum.bindings.custatevec import SamplerOutput
        from pytket.utils.outcomearray import OutcomeArray

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

                sampler_descriptor, size_t = cusv.sampler_create(
                    libhandle.handle,
                    sv.array.data.ptr,
                    cudaDataType.CUDA_C_64F,
                    sv.n_qubits,
                    n_shots,
                )

                bit_strings = np.empty(n_shots, dtype=np.int32)
                # LSB-MSB ordering of the bit strings
                bit_ordering = sorted([i.index[0] for i in circuit.qubits])
                rng = np.random.default_rng(seed)
                randnums = rng.random(n_shots, dtype=np.float64).tolist()
                
                cusv.sampler_preprocess(
                    handle=libhandle.handle,
                    sampler=sampler_descriptor,
                    extra_workspace=0,
                    extra_workspace_size_in_bytes=0,
                    )

                cusv.sampler_sample(handle=libhandle.handle,
                                    sampler=sampler_descriptor,
                                    bit_strings=bit_strings.ctypes.data,
                                    bit_ordering=bit_ordering,
                                    bit_string_len=len(circuit.qubits),
                                    randnums=randnums,
                                    n_shots=n_shots,
                                    output=SamplerOutput.RANDNUM_ORDER,
                                    )

                cusv.sampler_destroy(sampler_descriptor)
                #TODO: Postprocessing - sample only the qubits that should be measured?

        handle = ResultHandle(str(uuid4()))
        # In order to be able to use the BackendResult functionality,
        # we only pass the array of the statevector to BackendResult
        self._cache[handle] = {"result": BackendResult(state=cp.asnumpy(sv.array),
                                                       shots=OutcomeArray.from_readouts(bit_strings))}
        handle_list.append(handle)
        return handle_list


def _check_all_unitary_or_measurements(circuit: Circuit) -> bool:
    """Auxiliary function for custom predicate"""
    try:
        for cmd in circuit:
            if cmd.op.type != OpType.Measure:
                cmd.op.get_unitary()
        return True
    except:
        return False
