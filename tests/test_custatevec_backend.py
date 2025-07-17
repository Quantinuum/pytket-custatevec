import cupy as cp
import numpy as np
import pytest

from pytket.extensions.custatevec.backends import (
    CuStateVecShotsBackend,
    CuStateVecStateBackend,
)
from pytket.extensions.qiskit.backends.aer import AerStateBackend
from pytket.extensions.qulacs.backends import QulacsBackend
from pytket.utils import get_operator_expectation_value


@pytest.mark.parametrize(
    "statevector_circuit_fixture",
    [
        "test_circuit",
        "bell_circuit",
        "three_qubit_ghz_circuit",
        "four_qubit_superposition_circuit",
        "single_qubit_clifford_circuit",
        "single_qubit_non_clifford_circuit",
        "two_qubit_entangling_circuit",
        "global_phase_circuit",
    ],
)
def test_custatevecstate_vs_aer_and_qulacs(
    statevector_circuit_fixture: str, request: pytest.FixtureRequest,
) -> None:
    """Test the CuStateVecStateBackend against AerState and Qulacs Backends for various quantum circuits.

    Args:
        circuit_fixture: The fixture name for the quantum circuit to test.
        request: The pytest request object to access the fixture.

    Returns:
        None
    Compares the resulting quantum states to ensure consistency.
    """
    circuit_data = request.getfixturevalue(statevector_circuit_fixture)
    if isinstance(circuit_data, tuple):
        circuit, expected = circuit_data
    else:
        circuit = circuit_data
        expected = None

    cu_backend = CuStateVecStateBackend()
    cu_circuit = cu_backend.get_compiled_circuit(circuit)
    cu_handle = cu_backend.process_circuits([cu_circuit])
    cu_result = cp.asnumpy(cu_backend.get_result(cu_handle[0]).get_state().array)

    if expected is not None:
        assert np.allclose(cu_result, expected)
    else:
        # Test against Qulacs Backend
        qulacs_backend = QulacsBackend()
        qulacs_circuit = qulacs_backend.get_compiled_circuit(circuit)
        qulacs_handle = qulacs_backend.process_circuit(qulacs_circuit)
        qulacs_result = qulacs_backend.get_result(qulacs_handle).get_state()
        assert np.allclose(cu_result, qulacs_result)

        # Test against AerState Backend
        aer_backend = AerStateBackend()
        aer_circuit = aer_backend.get_compiled_circuit(circuit)
        aer_handle = aer_backend.process_circuit(aer_circuit)
        aer_result = aer_backend.get_result(aer_handle).get_state()
        assert np.allclose(cu_result, aer_result)

        # Test against pytket
        pytket_result = circuit.get_statevector()
        assert np.allclose(cu_result, pytket_result)


@pytest.mark.parametrize(
    "sampler_circuit_fixture, operator_fixture",
    [
        # ("bell_circuit", "bell_operator"),
        # ("three_qubit_ghz_circuit", "ghz_operator"),
        # ("four_qubit_superposition_circuit", "superposition_operator"),
        ("two_qubit_entangling_circuit", "entangling_operator"),
    ],
)
def test_custatevecshots_vs_aer_and_qulacs(
    sampler_circuit_fixture: str, operator_fixture: str, request: pytest.FixtureRequest,
) -> None:
    """Test the CuStateVecShotsBackend against AerState and Qulacs Backends for various quantum circuits.

    Args:
        sampler_circuit_fixture: The fixture name for the quantum circuit to test.
        operator_fixture: The fixture name for the operator to test.
        request: The pytest request object to access the fixtures.

    Returns:
        None
    """
    circuit_data = request.getfixturevalue(sampler_circuit_fixture)
    if isinstance(circuit_data, tuple):
        circuit = circuit_data[0]  # Extract the Circuit object
    else:
        circuit = circuit_data

    operator = request.getfixturevalue(operator_fixture)

    # CuStateVec expectation value
    cu_backend = CuStateVecShotsBackend()
    n_shots = 100000
    cu_circuit = cu_backend.get_compiled_circuit(circuit)
    cu_expectation = get_operator_expectation_value(
        cu_circuit, operator, cu_backend, n_shots,
    )

    # Qulacs expectation value
    qulacs_backend = QulacsBackend()
    qulacs_circuit = qulacs_backend.get_compiled_circuit(circuit)
    qulacs_expectation = get_operator_expectation_value(
        qulacs_circuit, operator, qulacs_backend,
    )

    assert np.isclose(cu_expectation, qulacs_expectation, atol=0.01)

    # AerState expectation value
    aer_backend = AerStateBackend()
    aer_circuit = aer_backend.get_compiled_circuit(circuit)
    aer_expectation = get_operator_expectation_value(aer_circuit, operator, aer_backend)

    assert np.isclose(cu_expectation, aer_expectation, atol=0.01)


def test_initial_statevector():
    """Test the initial_statevector function for all possible types and different qubit numbers and compare against the expected state vector."""
    from cuquantum.bindings._utils import cudaDataType

    from pytket.extensions.custatevec.custatevec import initial_statevector
    from pytket.extensions.custatevec.handle import CuStateVecHandle

    initial_states = {
        "zero": lambda n: np.eye(1, 2**n, 0, dtype=np.complex128).ravel(),
        "uniform": lambda n: np.full(2**n, 1 / np.sqrt(2**n), dtype=np.complex128),
        "ghz": lambda n: np.array(
            [1 / np.sqrt(2) if i in [0, 2**n - 1] else 0 for i in range(2**n)],
            dtype=np.complex128,
        ),
        "w": lambda n: np.array(
            [1 / np.sqrt(n) if (i).bit_count() == 1 else 0 for i in range(2**n)],
            dtype=np.complex128,
        ),
    }

    qubit_numbers = [2, 3, 4]

    for state_name, state_func in initial_states.items():
        for n in qubit_numbers:
            with CuStateVecHandle() as libhandle:
                sv = initial_statevector(
                    libhandle,
                    n,
                    state_name,
                    dtype=cudaDataType.CUDA_C_64F,
                )
                generated_state = sv.array
                expected_state = state_func(n)
                assert np.allclose(
                    generated_state, expected_state,
                ), f"Mismatch for {state_name} with {n} qubits"
