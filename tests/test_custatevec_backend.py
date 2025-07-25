import numpy as np
import pytest

from pytket.circuit import BasisOrder, Circuit
from pytket.extensions.custatevec.backends import (
    CuStateVecShotsBackend,
    CuStateVecStateBackend,
)
from pytket.extensions.qiskit.backends.aer import AerStateBackend, AerBackend
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
def test_custatevecstate_state_vector_vs_aer_and_qulacs(
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
    cu_result = cu_backend.get_result(cu_handle[0]).get_state()

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
        ("bell_circuit", "bell_operator"),
        ("three_qubit_ghz_circuit", "ghz_operator"),
        ("four_qubit_superposition_circuit", "superposition_operator"),
        ("two_qubit_entangling_circuit", "entangling_operator"),
    ],
)
def test_custatevecstate_expectation_value_vs_aer_and_qulacs(
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
    cu_backend = CuStateVecStateBackend()
    cu_circuit = cu_backend.get_compiled_circuit(circuit)
    cu_handle = cu_backend.run_circuit(cu_circuit)
    state = cu_handle.get_state()
    # Alternatively, use the get_operator_expectation_value function
    cu_expectation = get_operator_expectation_value(circuit, operator, cu_backend)

    # NOTE: The expectation values can be computed in general in two different ways
    # 1. Using the operator.state_expectation method
    # 2. Using pytket's default get_operator_expectation_value function with the non-compiled circuit
    # or add circuit.replace_implicit_wire_swaps() in case one wants to use the compiled circuit.

    # We defined a backend-specific get_operator_expectation_value method here 
    # to take advantage of CuStateVec's functionalities.

    assert np.allclose(operator.state_expectation(state), cu_expectation)

    # Qulacs expectation value
    qulacs_backend = QulacsBackend()
    qulacs_circuit = qulacs_backend.get_compiled_circuit(circuit)
    qulacs_handle = qulacs_backend.process_circuit(qulacs_circuit)
    qulacs_state = qulacs_backend.get_result(qulacs_handle).get_state()
    assert np.allclose(operator.state_expectation(qulacs_state), cu_expectation)

    # AerState expectation value
    aer_backend = AerStateBackend()
    aer_circuit = aer_backend.get_compiled_circuit(circuit)
    aer_handle = aer_backend.process_circuit(aer_circuit)
    aer_state = aer_backend.get_result(aer_handle).get_state()
    assert np.allclose(operator.state_expectation(aer_state), cu_expectation)

def test_basisorder() -> None:
    """Test the basis order of the CuStateVecShotsBackend."""
    from pytket.circuit import BasisOrder, Circuit
    c = Circuit(2)
    c.X(1)

    cu_backend = CuStateVecStateBackend()
    c = cu_backend.get_compiled_circuit(c)
    cu_handle = cu_backend.process_circuits([c])
    cu_result = cu_backend.get_result(cu_handle[0])
    assert np.allclose(cu_result.get_state(), np.asarray([0, 1, 0, 0]))
    assert np.allclose(cu_result.get_state(basis=BasisOrder.dlo), np.asarray([0, 0, 1, 0]))


def test_implicit_perm() -> None:
    """Test the implicit qubit permutation in CuStateVecStateBackend."""
    from pytket.passes import CliffordSimp
    c = Circuit(2)
    c.CX(0, 1)
    c.CX(1, 0)
    c.Ry(0.1, 1)
    c1 = c.copy()
    CliffordSimp().apply(c1)
    b = CuStateVecStateBackend()
    c = b.get_compiled_circuit(c, optimisation_level=1)
    c1 = b.get_compiled_circuit(c1, optimisation_level=1)
    assert c.implicit_qubit_permutation() != c1.implicit_qubit_permutation()
    h, h1 = b.process_circuits([c, c1])
    r, r1 = b.get_results([h, h1])
    for bo in [BasisOrder.ilo, BasisOrder.dlo]:
        s = r.get_state(basis=bo)
        s1 = r1.get_state(basis=bo)
        assert np.allclose(s, s1)

# ====================================
# === TESTS FOR SHOT-BASED BACKEND ===
# ====================================

def test_sampler_bell() -> None:
    n_shots = 1000
    c = Circuit(2, 2)
    c.H(0)
    c.CX(0, 1)
    c.measure_all()
    cu_backend = CuStateVecShotsBackend()
    c = cu_backend.get_compiled_circuit(c)
    cu_handle = cu_backend.process_circuit(c, n_shots=n_shots, seed=3)
    cu_result = cu_backend.get_result(cu_handle)
    assert cu_result.get_shots().shape == (n_shots, 2)
    
    counts = cu_result.get_counts()
    ratio = counts[(0, 0)] / counts[(1, 1)]
    assert np.isclose(ratio, 1, atol=0.2)

def test_sampler_basisorder() -> None:
    c = Circuit(2, 2)
    c.X(1)
    c.measure_all()
    cu_backend = CuStateVecShotsBackend()
    c = cu_backend.get_compiled_circuit(c)
    res = cu_backend.run_circuit(c, n_shots=10)
    assert res.get_counts() == {(0, 1): 10}
    assert res.get_counts(basis=BasisOrder.dlo) == {(1, 0): 10}

def test_sampler_expectation_value() -> None:
    from pytket.pauli import Pauli, QubitPauliString
    from pytket.utils.operators import QubitPauliOperator
    from pytket.circuit import Circuit, Qubit
    c = Circuit(2)
    c.H(0)
    c.CX(0, 1)
    op = QubitPauliOperator(
        {
            QubitPauliString({Qubit(0): Pauli.Z, Qubit(1): Pauli.Z}): 1.0,
            QubitPauliString({Qubit(0): Pauli.X, Qubit(1): Pauli.X}): 0.3,
            QubitPauliString({Qubit(0): Pauli.Z, Qubit(1): Pauli.Y}): 0.8j,
            QubitPauliString({Qubit(0): Pauli.Y}): -0.4j,
        }
    )
    b = CuStateVecShotsBackend()
    c = b.get_compiled_circuit(c)
    expectation = get_operator_expectation_value(c, op, b, n_shots=2000, seed=0)
    assert (np.real(expectation), np.imag(expectation)) == pytest.approx(
        (1.3, 0.0), abs=0.1
    )


@pytest.mark.parametrize(
    "sampler_circuit_fixture, operator_fixture",
    [
        ("bell_circuit", "bell_operator"),
        # ("three_qubit_ghz_circuit", "ghz_operator"),
        # ("four_qubit_superposition_circuit", "superposition_operator"),
        # ("two_qubit_entangling_circuit", "entangling_operator"),
    ],
)
def test_custatevecshots_expectation_value_vs_aer_and_qulacs(
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
    cu_expectation = get_operator_expectation_value(circuit, operator, cu_backend, n_shots)

    # AerState expectation value
    aer_backend = AerBackend()
    aer_expectation = get_operator_expectation_value(circuit, operator, aer_backend, n_shots)
    assert np.isclose(cu_expectation, aer_expectation, atol=0.1)

    # Qulacs expectation value
    qulacs_backend = QulacsBackend()
    qulacs_expectation = get_operator_expectation_value(circuit, operator, qulacs_backend)
    assert np.isclose(cu_expectation, qulacs_expectation, atol=0.1)

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
