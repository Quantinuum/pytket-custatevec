import pytest
import numpy as np
from pytket.circuit import Circuit, Qubit
from pytket.pauli import Pauli, QubitPauliString
from pytket.utils.operators import QubitPauliOperator

# Group 1: Single-Qubit Clifford Gates
@pytest.fixture
def single_qubit_clifford_circuit():
    c = Circuit(3, 3)
    c.X(0)
    c.Y(1)
    c.Z(2)
    c.H(0)
    c.S(1)
    c.SX(2)
    c.V(0)
    return c

@pytest.fixture
def bell_circuit():
    c = Circuit(2, 2)
    c.H(0)
    c.CX(0, 1)
    expected = np.asarray([1, 0, 0, 1]) * 1 / np.sqrt(2)
    return c, expected

@pytest.fixture
def test_circuit():
    c = Circuit(3, 3)
    c.X(2)
    c.H(0)
    c.CX(0, 1)
    return c

@pytest.fixture
def three_qubit_ghz_circuit():
    c = Circuit(3, 3)
    c.H(0)
    c.CX(0, 1)
    c.CX(1, 2)
    expected = np.asarray([1, 0, 0, 0, 0, 0, 0, 1]) * 1 / np.sqrt(2)
    return c, expected

@pytest.fixture
def four_qubit_superposition_circuit():
    c = Circuit(4, 4)
    for i in range(4):
        c.H(i)
    expected = np.ones(16) / 4
    return c, expected

# Group 2: Single-Qubit Non-Clifford or Parameterized Gates
@pytest.fixture
def single_qubit_non_clifford_circuit():
    c = Circuit(3, 3)
    c.T(0)
    c.Rx(0.5, 1)
    c.Ry(0.7, 2)
    c.Rz(0.9, 0)
    c.U1(0.3, 1)
    c.U2(0.4, 0.6, 2)
    c.U3(0.5, 0.7, 0.9, 0)
    c.PhasedX(0.2, 0.4, 1)
    return c

# Group 3: Two-Qubit Entangling Gates
@pytest.fixture
def two_qubit_entangling_circuit():
    c = Circuit(4, 4)
    c.ECR(0, 1)
    c.SWAP(1, 2)
    c.ISWAP(0.5, 2, 3)
    c.PhasedISWAP(0.3, 0.7, 3, 0)
    c.XXPhase(0.5, 0, 1)
    c.YYPhase(0.6, 2, 3)
    c.ZZPhase(0.4, 1, 2)
    c.ZZMax(0, 3)
    return c

# Group 4: Miscellaneous Circuits
@pytest.fixture
def global_phase_circuit():
    c = Circuit(1, 1)
    c.add_phase(0.5)
    expected = np.asarray([1, 0]) * np.exp(1j * np.pi * 0.5)
    return c, expected

# Operator Fixtures
@pytest.fixture
def bell_operator():
    """Fixture for a sample operator to test with the Bell circuit."""
    return QubitPauliOperator(
        {
            QubitPauliString({Qubit(0): Pauli.Z, Qubit(1): Pauli.Z}): 1.0,
            QubitPauliString({Qubit(0): Pauli.X, Qubit(1): Pauli.X}): 0.3,
            QubitPauliString({Qubit(0): Pauli.Z, Qubit(1): Pauli.Y}): 0.8j,
            QubitPauliString({Qubit(0): Pauli.Y}): -0.4j,
        }
    )

@pytest.fixture
def ghz_operator():
    """Fixture for an operator to test with the GHZ circuit."""
    return QubitPauliOperator(
        {
            QubitPauliString({Qubit(0): Pauli.Z, Qubit(1): Pauli.Z, Qubit(2): Pauli.Z}): 1.0,
            QubitPauliString({Qubit(0): Pauli.X, Qubit(1): Pauli.X, Qubit(2): Pauli.X}): 0.5,
        }
    )

@pytest.fixture
def superposition_operator():
    """Fixture for an operator to test with the superposition circuit."""
    return QubitPauliOperator(
        {
            QubitPauliString({Qubit(0): Pauli.Z}): 0.7,
            QubitPauliString({Qubit(1): Pauli.Z}): 0.7,
            QubitPauliString({Qubit(2): Pauli.Z}): 0.7,
            QubitPauliString({Qubit(3): Pauli.Z}): 0.7,
        }
    )

@pytest.fixture
def entangling_operator():
    """Fixture for an operator to test with the entangling circuit."""
    return QubitPauliOperator(
        {
            QubitPauliString({Qubit(0): Pauli.X, Qubit(1): Pauli.X}): 0.8,
            QubitPauliString({Qubit(2): Pauli.Y, Qubit(3): Pauli.Y}): 0.6,
        }
    )

