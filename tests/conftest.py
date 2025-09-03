import numpy as np  # noqa: D100
import pytest

from pytket._tket.circuit import Circuit
from pytket.circuit import Qubit
from pytket.pauli import Pauli, QubitPauliString
from pytket.utils.operators import QubitPauliOperator


# Group 1: Single-Qubit Clifford Gates
@pytest.fixture
def single_qubit_clifford_circuit() -> Circuit:
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
def bell_circuit() -> tuple[Circuit, np.ndarray]:
    c = Circuit(2, 2)
    c.H(0)
    c.CX(0, 1)
    expected = np.asarray([1, 0, 0, 1]) * 1 / np.sqrt(2)
    return c, expected


@pytest.fixture
def test_circuit() -> Circuit:
    c = Circuit(3, 3)
    c.X(2)
    c.H(0)
    c.CX(0, 1)
    return c


@pytest.fixture
def three_qubit_ghz_circuit() -> tuple[Circuit, np.ndarray]:
    c = Circuit(3, 3)
    c.H(0)
    c.CX(0, 1)
    c.CX(1, 2)
    expected = np.asarray([1, 0, 0, 0, 0, 0, 0, 1]) * 1 / np.sqrt(2)
    return c, expected


@pytest.fixture
def four_qubit_superposition_circuit() -> tuple[Circuit, np.ndarray]:
    c = Circuit(4, 4)
    for i in range(4):
        c.H(i)
    expected = np.ones(16) / 4
    return c, expected


# Group 2: Single-Qubit Non-Clifford or Parameterized Gates
@pytest.fixture
def single_qubit_non_clifford_circuit() -> Circuit:
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
def two_qubit_entangling_circuit() -> Circuit:
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
def global_phase_circuit() -> tuple[Circuit, np.ndarray]:
    c = Circuit(1, 1)
    c.add_phase(0.5)
    expected = np.asarray([1, 0]) * np.exp(1j * np.pi * 0.5)
    return c, expected


@pytest.fixture
def circuit_with_adjoint_gates() -> Circuit:
    """Circuit with adjoint gates."""
    c = Circuit(3, 3)
    c.H(0)
    c.Sdg(1)
    c.Tdg(2)
    c.CSdg(0, 1)
    c.Vdg(0)
    c.SXdg(2)
    c.Vdg(2)
    c.CSXdg(1, 2)
    return c


def random_line_circuit(n_qubits: int, layers: int) -> Circuit:
    """Random circuit with line connectivity."""
    c = Circuit(n_qubits)

    for i in range(layers):
        # Layer of TK1 gates
        for q in range(n_qubits):
            c.TK1(np.random.rand(), np.random.rand(), np.random.rand(), q)  # noqa: NPY002

        # Layer of CX gates
        offset = np.mod(i, 2)  # Even layers connect (q0,q1), odd (q1,q2)
        qubit_pairs = [[c.qubits[i], c.qubits[i + 1]] for i in range(offset, n_qubits - 1, 2)]
        # Direction of each CX gate is random
        for pair in qubit_pairs:
            if np.random.rand() > 0.5:  # noqa: NPY002, PLR2004
                pair = [pair[1], pair[0]]  # noqa: PLW2901

        for pair in qubit_pairs:
            c.CX(pair[0], pair[1])

    return c


# def quantum_volume_circuit(n_qubits: int) -> Circuit:
#     """Random quantum volume circuit."""
#     depth = n_qubits
#     c = Circuit(n_qubits)

#     for _ in range(depth):
#         qubits = np.random.permutation([i for i in range(n_qubits)])  # noqa: C416, NPY002
#         qubit_pairs = [[qubits[i], qubits[i + 1]] for i in range(0, n_qubits - 1, 2)]

#         for pair in qubit_pairs:
#             # Generate random 4x4 unitary matrix.
#             SU4 = unitary_group.rvs(4)  # random unitary in SU4
#             SU4 = SU4 / (np.linalg.det(SU4) ** 0.25)
#             SU4 = np.asarray(SU4)
#             c.add_unitary2qbox(Unitary2qBox(SU4), *pair)

#     DecomposeBoxes().apply(c)
#     return c


@pytest.fixture
def q1_empty() -> Circuit:
    circuit = Circuit(1)
    return circuit  # noqa: RET504


@pytest.fixture
def q5_empty() -> Circuit:
    circuit = Circuit(5)
    return circuit  # noqa: RET504


@pytest.fixture
def q8_empty() -> Circuit:
    circuit = Circuit(8)
    return circuit  # noqa: RET504


@pytest.fixture
def q1_h0rz() -> Circuit:
    circuit = Circuit(1)
    circuit.H(0).Rz(0.3, 0)
    return circuit


@pytest.fixture
def q2_x0() -> Circuit:
    circuit = Circuit(2)
    circuit.X(0)
    return circuit


@pytest.fixture
def q2_x1() -> Circuit:
    circuit = Circuit(2)
    circuit.X(1)
    return circuit


@pytest.fixture
def q2_v0() -> Circuit:
    circuit = Circuit(2)
    circuit.V(0)
    return circuit


@pytest.fixture
def q2_x0cx01() -> Circuit:
    circuit = Circuit(2)
    circuit.X(0).CX(0, 1)
    return circuit


@pytest.fixture
def q2_x1cx10x1() -> Circuit:
    circuit = Circuit(2)
    circuit.X(1).CX(1, 0).X(1)
    return circuit


@pytest.fixture
def q2_x0cx01cx10() -> Circuit:
    circuit = Circuit(2)
    circuit.X(0).CX(0, 1).CX(1, 0)
    return circuit


@pytest.fixture
def q2_v0cx01cx10() -> Circuit:
    circuit = Circuit(2)
    circuit.V(0).CX(0, 1).CX(1, 0)
    return circuit


@pytest.fixture
def q2_hadamard_test() -> Circuit:
    circuit = Circuit(2)
    circuit.H(0).CRx(0.5, 0, 1).H(0)
    return circuit


@pytest.fixture
def q2_lcu1() -> Circuit:
    circuit = Circuit(2)
    circuit.Ry(0.78, 1).Ry(0.27, 0).CX(0, 1).CZ(0, 1).Ry(-0.27, 0)
    return circuit


@pytest.fixture
def q2_lcu2() -> Circuit:
    circuit = Circuit(2)
    circuit.Ry(0.78, 1).Ry(0.27, 0).CZ(0, 1).CY(0, 1).Ry(-0.27, 0)
    return circuit


@pytest.fixture
def q2_lcu3() -> Circuit:
    circuit = Circuit(2)
    circuit.Ry(0.78, 1).Rx(0.67, 0).CX(0, 1).CZ(0, 1).Ry(-0.67, 0)
    return circuit


@pytest.fixture
def q3_v0cx02() -> Circuit:
    circuit = Circuit(3)
    circuit.V(0).CX(0, 2)
    return circuit


@pytest.fixture
def q3_cx01cz12x1rx0() -> Circuit:
    circuit = Circuit(3)
    circuit.CX(0, 1).CZ(1, 2).X(1).Rx(0.3, 0)
    return circuit


# @pytest.fixture
# def q4_lcu1() -> Circuit:
#     circuit = Circuit(4)
#     circuit.Ry(0.78, 3).Ry(0.27, 2).CX(2, 3).Ry(0.58, 2).Ry(0.21, 3)
#     circuit.Ry(0.12, 0).Ry(0.56, 1)
#     circuit.add_gate(OpType.CnX, [0, 1, 2]).add_gate(OpType.CnX, [0, 1, 3])
#     circuit.X(0).X(1).add_gate(OpType.CnY, [0, 1, 2]).add_gate(OpType.CnY, [0, 1, 3]).X(
#         0,
#     ).X(1)
#     circuit.Ry(-0.12, 0).Ry(-0.56, 1)
#     return circuit


# @pytest.fixture
# def q4_lcu1_parameterised() -> Circuit:
#     a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
#     circuit = Circuit(4)
#     circuit.Ry(a, 3).Ry(0.27, 2).CX(2, 3).Ry(b, 2).Ry(0.21, 3)
#     circuit.Ry(0.12, 0).Ry(a, 1)
#     circuit.add_gate(OpType.CnX, [0, 1, 2]).add_gate(OpType.CnX, [0, 1, 3])
#     circuit.X(0).X(1).add_gate(OpType.CnY, [0, 1, 2]).add_gate(OpType.CnY, [0, 1, 3]).X(
#         0,
#     ).X(1)
#     circuit.Ry(-b, 0).Ry(-c, 1)
#     return circuit


@pytest.fixture
def q4_multicontrols() -> Circuit:
    circ = Circuit(4)
    circ.X(0)
    circ.X(1)
    circ.X(2)
    circ.X(3)
    circ.CCX(0, 1, 2)
    circ.CCX(0, 1, 3)
    circ.CSWAP(0, 1, 2)
    circ.CCX(0, 2, 3)
    circ.CSWAP(3, 1, 0)
    circ.CCX(3, 2, 1)
    circ.CCX(3, 2, 0)
    circ.X(1)
    circ.CCX(3, 1, 0)
    return circ


@pytest.fixture
def q4_with_creates() -> Circuit:
    circuit = Circuit(4)
    circuit.qubit_create_all()

    circuit.S(1)
    circuit.Rz(0.3, 0)
    circuit.Ry(0.1, 2)
    circuit.TK1(0.2, 0.9, 0.8, 3)
    circuit.TK2(0.6, 0.5, 0.7, 1, 2)
    circuit.X(0)
    circuit.H(2)
    circuit.V(1)
    circuit.Z(3)

    return circuit


@pytest.fixture
def q5_h0s1rz2ry3tk4tk13() -> Circuit:
    circuit = Circuit(5)
    circuit.H(0)
    circuit.S(1)
    circuit.Rz(0.3, 2)
    circuit.Ry(0.1, 3)
    circuit.TK1(0.2, 0.9, 0.8, 4)
    circuit.TK2(0.6, 0.5, 0.7, 1, 3)
    return circuit


# @pytest.fixture
# def q5_h0s1rz2ry3tk4tk13_parameterised() -> Circuit:
#     a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
#     circuit = Circuit(5)
#     circuit.H(0)
#     circuit.S(1)
#     circuit.Rz(a * c, 2)
#     circuit.Ry(b + a, 3)
#     circuit.TK1(a, b, c, 4)
#     circuit.TK2(a - b, c - a, (a + b) * c, 1, 3)
#     return circuit


@pytest.fixture
def q8_x0h2v5z6() -> Circuit:
    circuit = Circuit(8)
    circuit.X(0)
    circuit.H(2)
    circuit.V(5)
    circuit.Z(6)
    return circuit


@pytest.fixture
def q5_line_circ_30_layers() -> Circuit:
    np.random.seed(1)  # noqa: NPY002
    return random_line_circuit(n_qubits=5, layers=30)


@pytest.fixture
def q20_line_circ_20_layers() -> Circuit:
    np.random.seed(1)  # noqa: NPY002
    return random_line_circuit(n_qubits=20, layers=20)


# @pytest.fixture
# def q6_qvol() -> Circuit:
#     np.random.seed(1)  # noqa: NPY002
#     return quantum_volume_circuit(n_qubits=6)


# @pytest.fixture
# def q8_qvol() -> Circuit:
#     np.random.seed(1)  # noqa: NPY002
#     return quantum_volume_circuit(n_qubits=8)


# @pytest.fixture
# def q15_qvol() -> Circuit:
#     np.random.seed(1)  # noqa: NPY002
#     return quantum_volume_circuit(n_qubits=15)


# @pytest.fixture
# def q3_toffoli_box_with_implicit_swaps() -> Circuit:
#     # Using specific permutation here
#     perm = {
#         (False, False): (True, True),
#         (False, True): (False, False),
#         (True, False): (True, False),
#         (True, True): (False, True),
#     }

#     # Create a circuit with more qubits and multiple applications of the permutation
#     # above
#     circ = Circuit(3)

#     # Create the circuit
#     circ.add_toffolibox(ToffoliBox(perm), [Qubit(0), Qubit(1)])
#     circ.add_toffolibox(ToffoliBox(perm), [Qubit(1), Qubit(2)])

#     DecomposeBoxes().apply(circ)
#     CnXPairwiseDecomposition().apply(circ)
#     Transform.OptimiseCliffords().apply(circ)

#     # Check that, indeed, there are implicit swaps
#     implicit_perm = circ.implicit_qubit_permutation()
#     assert any(qin != qout for qin, qout in implicit_perm.items())

#     return circ


# Operator Fixtures


@pytest.fixture
def single_qubit_operator() -> QubitPauliOperator:
    """Fixture for a sample operator to test with single-qubit circuits."""
    return QubitPauliOperator(
        {
            QubitPauliString({Qubit(0): Pauli.X}): 0.4,
        },
    )


@pytest.fixture
def two_qubit_operator() -> QubitPauliOperator:
    """Fixture for a sample operator to test with the Bell circuit."""
    return QubitPauliOperator(
        {
            QubitPauliString({Qubit(0): Pauli.Z, Qubit(1): Pauli.Z}): 1.0,
            QubitPauliString({Qubit(0): Pauli.X, Qubit(1): Pauli.X}): 1.0,
            QubitPauliString({Qubit(0): Pauli.Z, Qubit(1): Pauli.Y}): 0.8j,
            QubitPauliString({Qubit(0): Pauli.Y}): -0.4j,
        },
    )


@pytest.fixture
def three_qubit_operator() -> QubitPauliOperator:
    """Fixture for an operator to test with the GHZ circuit."""
    return QubitPauliOperator(
        {
            QubitPauliString({Qubit(0): Pauli.Z, Qubit(1): Pauli.Z, Qubit(2): Pauli.Z}): 1.0,
            QubitPauliString({Qubit(0): Pauli.X, Qubit(1): Pauli.X, Qubit(2): Pauli.X}): 0.5,
        },
    )


@pytest.fixture
def four_qubit_operator() -> QubitPauliOperator:
    """Fixture for an operator to test with the entangling circuit."""
    return QubitPauliOperator(
        {
            QubitPauliString({Qubit(0): Pauli.X, Qubit(1): Pauli.X}): 0.8,
            QubitPauliString({Qubit(2): Pauli.Y, Qubit(3): Pauli.Y}): 0.6,
        },
    )
