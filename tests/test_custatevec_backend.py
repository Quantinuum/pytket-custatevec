import numpy as np  # noqa: D100
import pytest
from cuquantum.bindings._utils import cudaDataType

from pytket._tket.circuit import Circuit
from pytket.circuit import BasisOrder
from pytket.extensions.custatevec.backends import (
    CuStateVecShotsBackend,
    CuStateVecStateBackend,
)
from pytket.extensions.custatevec.custatevec import initial_statevector
from pytket.extensions.custatevec.handle import CuStateVecHandle
from pytket.extensions.qiskit.backends.aer import AerStateBackend
from pytket.extensions.qulacs.backends.qulacs_backend import QulacsBackend
from pytket.passes import CliffordSimp
from pytket.utils.expectations import get_operator_expectation_value

# =====================================================
# === TESTS FOR STATEVECTOR AND SHOT-BASED BACKENDS ===
# =====================================================

def test_initial_statevector() -> None:
    """Test the initial_statevector function for all possible types and different qubit numbers.

    Compare against the expected state vector.

    Notes:
        - Since the statevectors/amplitude arrays of all possible initial states {zero, uniform, ghz, w}
        are "closed under reversal", i.e. if one reverses any computational basis state
        one gets another computational basis state also present in the statevector,
        all resulting amplitude arrays will be identical for little endian and big endian order.
        For example states like |ψ⟩= 1/√2(|00⟩+|11⟩) always correspond to [1/√2, 0, 0, 1/√2].

    """
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
                    state_name, # type: ignore[no-untyped-call]
                    dtype=cudaDataType.CUDA_C_64F,
                )
                generated_state = sv.array
                expected_state = state_func(n)
                assert np.allclose(
                    generated_state, expected_state,
                ), f"Mismatch for {state_name} with {n} qubits"


# =====================================
# === TESTS FOR STATEVECTOR BACKEND ===
# =====================================

# TODO: Need to add more gates to test all circuits of cuTensorNet
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
        "circuit_with_adjoint_gates",
        "q1_empty",
        "q5_empty",
        "q8_empty",
        "q1_h0rz",
        "q2_x0",
        "q2_x1",
        "q2_v0",
        "q2_x0cx01",
        "q2_x1cx10x1",
        "q2_x0cx01cx10",
        "q2_v0cx01cx10",
        "q2_hadamard_test",
        "q2_lcu1",
        "q2_lcu2",
        "q2_lcu3",
        "q3_v0cx02",
        "q3_cx01cz12x1rx0",
        # "q4_lcu1",
        # "q4_lcu1_parameterised",
        "q4_multicontrols",
        "q4_with_creates",
        "q5_h0s1rz2ry3tk4tk13",
        # "q5_h0s1rz2ry3tk4tk13_parameterised",
        "q8_x0h2v5z6",
        "q5_line_circ_30_layers",
        # "q20_line_circ_20_layers",
        # "q6_qvol",
        # "q8_qvol",
        # "q15_qvol",
        # "q3_toffoli_box_with_implicit_swaps",
    ],
)
def test_custatevecstate_state_vector_vs_aer_and_qulacs(
    statevector_circuit_fixture: str, request: pytest.FixtureRequest,
) -> None:
    """Test the CuStateVecStateBackend against AerState and Qulacs Backends for various quantum circuits.

    Args:
        statevector_circuit_fixture: The fixture name for the quantum circuit to test.
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
    cu_handle = cu_backend.process_circuits(cu_circuit)
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
    ("statevector_circuit_fixture", "operator_fixture"),
    [
        ("test_circuit", "two_qubit_operator"),
        ("bell_circuit", "two_qubit_operator"),
        ("three_qubit_ghz_circuit", "three_qubit_operator"),
        ("four_qubit_superposition_circuit", "four_qubit_operator"),
        ("single_qubit_clifford_circuit", "two_qubit_operator"),
        ("single_qubit_non_clifford_circuit", "two_qubit_operator"),
        ("two_qubit_entangling_circuit", "two_qubit_operator"),
        ("global_phase_circuit", "single_qubit_operator"),
        ("circuit_with_adjoint_gates", "three_qubit_operator"),
        ("q1_empty", "single_qubit_operator"),
        ("q5_empty", "three_qubit_operator"),
        ("q8_empty", "four_qubit_operator"),
        ("q1_h0rz", "single_qubit_operator"),
        ("q2_x0", "two_qubit_operator"),
        ("q2_x1", "two_qubit_operator"),
        ("q2_v0", "two_qubit_operator"),
        ("q2_x0cx01", "two_qubit_operator"),
        ("q2_x1cx10x1", "two_qubit_operator"),
        ("q2_x0cx01cx10", "two_qubit_operator"),
        ("q2_v0cx01cx10", "two_qubit_operator"),
        # ("q2_hadamard_test", "two_qubit_operator"), #TODO: Add CrX gate
        ("q2_lcu1", "two_qubit_operator"),
        ("q2_lcu2", "two_qubit_operator"),
        ("q2_lcu3", "two_qubit_operator"),
        ("q3_v0cx02", "three_qubit_operator"),
        ("q3_cx01cz12x1rx0", "three_qubit_operator"),
        ("q4_multicontrols", "four_qubit_operator"),
        # ("q4_with_creates", "four_qubit_operator"), #TODO: Add TK2 gate
        # ("q5_h0s1rz2ry3tk4tk13", "three_qubit_operator"), #TODO: Add TK2 gate
        ("q8_x0h2v5z6", "four_qubit_operator"),
        ("q5_line_circ_30_layers", "three_qubit_operator"),
    ],
)
def test_custatevecstate_expectation_value_vs_aer_and_qulacs(
    statevector_circuit_fixture: str, operator_fixture: str, request: pytest.FixtureRequest,
) -> None:
    """Test the CuStateVecShotsBackend against AerState and Qulacs Backends for various quantum circuits.

    Args:
        statevector_circuit_fixture: The fixture name for the quantum circuit to test.
        operator_fixture: The fixture name for the operator to test.
        request: The pytest request object to access the fixtures.

    Returns:
        None
    """
    circuit_data = request.getfixturevalue(statevector_circuit_fixture)
    circuit = circuit_data[0] if isinstance(circuit_data, tuple) else circuit_data

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

def test_custatevecstate_basisorder() -> None:
    """Test the basis order of the CuStateVecStateBackend."""
    c = Circuit(2)
    c.X(1)

    cu_backend = CuStateVecStateBackend()
    c = cu_backend.get_compiled_circuit(c)
    cu_handle = cu_backend.process_circuits(c)
    cu_result = cu_backend.get_result(cu_handle[0])
    assert np.allclose(cu_result.get_state(), np.asarray([0, 1, 0, 0]))
    assert np.allclose(cu_result.get_state(basis=BasisOrder.dlo), np.asarray([0, 0, 1, 0]))

def test_custatevecstate_implicit_perm() -> None:
    """Test the implicit qubit permutation in CuStateVecStateBackend."""
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

@pytest.mark.parametrize(
    ("sampler_circuit_fixture", "operator_fixture"),
    [
        ("test_circuit", "two_qubit_operator"),
        ("bell_circuit", "two_qubit_operator"),
        ("three_qubit_ghz_circuit", "three_qubit_operator"),
        ("four_qubit_superposition_circuit", "four_qubit_operator"),
        ("single_qubit_clifford_circuit", "two_qubit_operator"),
        ("single_qubit_non_clifford_circuit", "two_qubit_operator"),
        ("two_qubit_entangling_circuit", "two_qubit_operator"),
        ("global_phase_circuit", "single_qubit_operator"),
        ("circuit_with_adjoint_gates", "three_qubit_operator"),
        ("q1_empty", "single_qubit_operator"),
        ("q5_empty", "three_qubit_operator"),
        ("q8_empty", "four_qubit_operator"),
        ("q1_h0rz", "single_qubit_operator"),
        ("q2_x0", "two_qubit_operator"),
        ("q2_x1", "two_qubit_operator"),
        ("q2_v0", "two_qubit_operator"),
        ("q2_x0cx01", "two_qubit_operator"),
        ("q2_x1cx10x1", "two_qubit_operator"),
        ("q2_x0cx01cx10", "two_qubit_operator"),
        ("q2_v0cx01cx10", "two_qubit_operator"),
        # ("q2_hadamard_test", "two_qubit_operator"), #TODO: Add CrX gate
        ("q2_lcu1", "two_qubit_operator"),
        ("q2_lcu2", "two_qubit_operator"),
        ("q2_lcu3", "two_qubit_operator"),
        ("q3_v0cx02", "three_qubit_operator"),
        ("q3_cx01cz12x1rx0", "three_qubit_operator"),
        ("q4_multicontrols", "four_qubit_operator"),
        # ("q4_with_creates", "four_qubit_operator"), #TODO: Add TK2 gate
        # ("q5_h0s1rz2ry3tk4tk13", "three_qubit_operator"), #TODO: Add TK2 gate
        ("q8_x0h2v5z6", "four_qubit_operator"),
        ("q5_line_circ_30_layers", "three_qubit_operator"),
    ],
)
def test_custatevecshots_expectation_value_vs_qulacs(
    sampler_circuit_fixture: str, operator_fixture: str, request: pytest.FixtureRequest,
) -> None:
    """Test the CuStateVecShotsBackend against Qulacs Backends for various quantum circuits.

    Args:
        sampler_circuit_fixture: The fixture name for the quantum circuit to test.
        operator_fixture: The fixture name for the operator to test.
        request: The pytest request object to access the fixtures.

    Returns:
        None
    """
    circuit_data = request.getfixturevalue(sampler_circuit_fixture)
    circuit = circuit_data[0] if isinstance(circuit_data, tuple) else circuit_data

    operator = request.getfixturevalue(operator_fixture)
    n_shots = 1000000
    # CuStateVec expectation value
    cu_backend = CuStateVecShotsBackend()
    cu_expectation = get_operator_expectation_value(circuit, operator, cu_backend, n_shots)

    # Qulacs expectation value
    qulacs_backend = QulacsBackend()
    qulacs_expectation = get_operator_expectation_value(circuit, operator, qulacs_backend, n_shots)

    assert np.isclose(cu_expectation, qulacs_expectation, atol=0.1)

def test_custatevecshots_basisorder() -> None:
    """Test the CuStateVecShotsBackend for basis order consistency in sampling."""
    c = Circuit(2, 2)
    c.X(1)
    c.measure_all()
    cu_backend = CuStateVecShotsBackend()
    c = cu_backend.get_compiled_circuit(c)
    cu_handle = cu_backend.process_circuits(c, n_shots=10)
    cu_result = cu_backend.get_result(cu_handle[0])
    assert cu_result.get_counts() == {(0, 1): 10}
    assert cu_result.get_counts(basis=BasisOrder.dlo) == {(1, 0): 10}

def test_custatevecshots_partial_measurement() -> None:
    """Test the CuStateVecShotsBackend with partial measurement."""
    circ = Circuit(3, 2)
    circ.Rx(0.3, 0).CX(0, 1).CZ(1, 2)
    circ.Measure(0, 0)
    circ.Measure(2, 1)
    cu_backend = CuStateVecShotsBackend()
    cu_circuit = cu_backend.get_compiled_circuit(circ)
    cu_handle = cu_backend.process_circuits(cu_circuit, n_shots=100)
    cu_result = cu_backend.get_result(cu_handle[0])
    cu_counts = cu_result.get_counts()

    qulacs_backend = QulacsBackend()
    qulacs_circuit = qulacs_backend.get_compiled_circuit(circ)
    qulacs_handle = qulacs_backend.process_circuit(qulacs_circuit, n_shots=100)
    qulacs_result = qulacs_backend.get_result(qulacs_handle)
    qulacs_counts = qulacs_result.get_counts()

    # Check that the readout qubits match for the measured qubits
    assert cu_counts.keys() == qulacs_counts.keys()
    for key in cu_counts:
        assert np.isclose(cu_counts[key], qulacs_counts[key], atol=20)
