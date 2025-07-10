import numpy as np
import pytest

from pytket.extensions.custatevec.backends import CuStateVecStateBackend


@pytest.mark.parametrize(
    "circuit_fixture",
    [
        "bell_circuit",
        "three_qubit_ghz_circuit",
        "four_qubit_superposition_circuit",
        "single_qubit_clifford_circuit",
        "single_qubit_non_clifford_circuit",
        "two_qubit_entangling_circuit",
    ],
)
def test_custatevec_vs_aer(circuit_fixture: str, request: pytest.FixtureRequest) -> None:
    """Test the CuStateVecStateBackend against the AerStateBackend for various quantum circuits.

    Args:
        circuit_fixture: The fixture name for the quantum circuit to test.
        request: The pytest request object to access the fixture.

    Returns:
        None
    Compares the resulting quantum states to ensure consistency.
    """
    circuit_data = request.getfixturevalue(circuit_fixture)
    if isinstance(circuit_data, tuple):
        circuit, expected = circuit_data
    else:
        circuit = circuit_data
        expected = None

    cu_backend = CuStateVecStateBackend()

    cu_handle = cu_backend.process_circuits([circuit])
    cu_result = cu_backend.get_result(cu_handle[0]).get_state().array

    if expected is not None:
        assert np.allclose(cu_result, expected)
    else:
        from pytket.extensions.qiskit.backends.aer import (  # noqa: PLC0415
            AerStateBackend,
        )
        aer_backend = AerStateBackend()
        if not aer_backend.valid_circuit(circuit):
            circuit = aer_backend.get_compiled_circuit(circuit)
        aer_handle = aer_backend.process_circuit(circuit)
        aer_result = aer_backend.get_result(aer_handle).get_state()
        assert np.allclose(cu_result, aer_result)
