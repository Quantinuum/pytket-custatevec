import numpy as np

from pytket.architecture import Architecture
from pytket.circuit import Circuit, Node
from pytket.extensions.custatevec.backends import CuStateVecStateBackend
from pytket.extensions.custatevec.utils import _remove_meas_and_implicit_swaps
from pytket.passes import DefaultMappingPass
from pytket.predicates import CompilationUnit


def test_remove_meas_and_implicit_swaps() -> None:
    """Test the _remove_meas_and_implicit_swaps function."""

    def circuit_with_measurement() -> Circuit:
        """Circuit with measurements.

        TODO: Enforce qubit architecture in the circuit to ensure that implicit swaps are present.
        """
        circ = Circuit(5)
        circ.CX(0, 1)
        circ.CX(0, 3)
        circ.CX(1, 4)
        circ.CX(0, 4)
        circ.measure_all()

        n = [Node("n", i) for i in range(5)]
        arc = Architecture([[n[0], n[1]], [n[1], n[2]], [n[2], n[3]], [n[3], n[4]]])
        mapper = DefaultMappingPass(arc)
        cu = CompilationUnit(circ)
        mapper.apply(cu)
        circ1 = cu.circuit

        return circ1

    original_circ = circuit_with_measurement()
    # Remove measurements
    clean_circ, _ = _remove_meas_and_implicit_swaps(original_circ)

    cu_backend = CuStateVecStateBackend()

    cu_handle_clean = cu_backend.process_circuits([clean_circ])
    sv_clean = cu_backend.get_result(cu_handle_clean[0]).get_state().array

    cu_handle_original = cu_backend.process_circuits([original_circ])
    sv_original = cu_backend.get_result(cu_handle_original[0]).get_state().array

    assert np.allclose(sv_original, sv_clean, atol=1e-8), "Statevectors do not match"

def test_remove_meas_and_implicit_swaps_mid_circuit() -> None:
    """Test the _remove_meas_and_implicit_swaps function with mid-circuit measurements.

    It should throw an error if the circuit has mid-circuit measurements.
    """
    def circuit_with_mid_circuit_measurements() -> Circuit:
        """Circuit with mid-circuit measurement."""
        circuit = Circuit(1, 1)
        circuit.H(0)
        circuit.Measure(circuit.qubits[0], circuit.bits[0])
        circuit.Reset(circuit.qubits[0])
        return circuit

    original_circ = circuit_with_mid_circuit_measurements()
    try:
        _remove_meas_and_implicit_swaps(original_circ)
    except ValueError as e:
        assert str(e) == "Circuit contains a mid-circuit measurement"  # noqa: PT017
    else:
        raise AssertionError("Expected ValueError for mid-circuit measurements not raised.")  # noqa: EM101, TRY003
