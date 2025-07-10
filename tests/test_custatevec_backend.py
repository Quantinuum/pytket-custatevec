# from typing import Any

import numpy as np
import pytest

from pytket.circuit import (
    Circuit,
)
from pytket.extensions.custatevec.backends import CuStateVecStateBackend
from pytket.extensions.qiskit import AerStateBackend


def test_bell() -> None:
    c = Circuit(2)
    c.H(0)
    c.CX(0, 1)
    cu_backend = CuStateVecStateBackend()
    cu_handle = cu_backend.process_circuits([c])
    cu_result = cu_backend.get_result(cu_handle[0]).get_state().array
    assert np.allclose(
        cu_result,
        np.asarray([1, 0, 0, 1]) * 1 / np.sqrt(2),
    )

def test_single_qubit_hadamard() -> None:
    c = Circuit(1)
    c.H(0)
    cu_backend = CuStateVecStateBackend()
    cu_handle = cu_backend.process_circuits([c])
    cu_result = cu_backend.get_result(cu_handle[0]).get_state().array
    assert np.allclose(
        cu_result,
        np.asarray([1, 1]) * 1 / np.sqrt(2),
    )

def test_three_qubit_ghz() -> None:
    c = Circuit(3)
    c.H(0)
    c.CX(0, 1)
    c.CX(1, 2)
    cu_backend = CuStateVecStateBackend()
    cu_handle = cu_backend.process_circuits([c])
    cu_result = cu_backend.get_result(cu_handle[0]).get_state().array
    assert np.allclose(
        cu_result,
        np.asarray([1, 0, 0, 0, 0, 0, 0, 1]) * 1 / np.sqrt(2),
    )

def test_four_qubit_superposition() -> None:
    c = Circuit(4)
    for i in range(4):
        c.H(i)
    cu_backend = CuStateVecStateBackend()
    cu_handle = cu_backend.process_circuits([c])
    cu_result = cu_backend.get_result(cu_handle[0]).get_state().array
    expected = np.ones(16) / 4  # Uniform superposition
    assert np.allclose(cu_result, expected)

def test_single_qubit_rotations() -> None:
    c = Circuit(1)
    c.Rx(0.5, 0)
    c.Ry(0.5, 0)
    c.Rz(0.5, 0)
    cu_backend = CuStateVecStateBackend()
    aer_backend = AerStateBackend()
    aer_handle = aer_backend.process_circuit(c)
    cu_handle = cu_backend.process_circuits([c])
    aer_result = aer_backend.get_result(aer_handle).get_state()
    cu_result = cu_backend.get_result(cu_handle[0]).get_state().array
    print(aer_result)
    print(cu_result)
    assert np.allclose(cu_result, aer_result)

def test_three_qubit_mixed_gates() -> None:
    c = Circuit(3)
    c.H(0)
    c.Rz(0.5, 1)
    c.CX(0, 1)
    c.Rx(0.25, 2)
    c.CX(1, 2)
    cu_backend = CuStateVecStateBackend()
    aer_backend = AerStateBackend()
    aer_handle = aer_backend.process_circuit(c)
    cu_handle = cu_backend.process_circuits([c])
    aer_result = aer_backend.get_result(aer_handle).get_state()
    cu_result = cu_backend.get_result(cu_handle[0]).get_state().array
    assert np.allclose(cu_result, aer_result)

def test_four_qubit_entanglement_and_rotation() -> None:
    c = Circuit(4)
    c.H(0)
    c.CX(0, 1)
    c.CX(1, 2)
    c.CX(2, 3)
    c.Ry(0.5, 3)
    cu_backend = CuStateVecStateBackend()
    aer_backend = AerStateBackend()
    aer_handle = aer_backend.process_circuit(c)
    cu_handle = cu_backend.process_circuits([c])
    aer_result = aer_backend.get_result(aer_handle).get_state()
    cu_result = cu_backend.get_result(cu_handle[0]).get_state().array
    assert np.allclose(cu_result, aer_result)

def test_five_qubit_complex_circuit() -> None:
    c = Circuit(5)
    c.H(0)
    c.CX(0, 1)
    c.Rz(0.3, 2)
    c.CX(1, 2)
    c.Rx(0.7, 3)
    c.CX(2, 3)
    c.Ry(0.2, 4)
    c.CX(3, 4)
    cu_backend = CuStateVecStateBackend()
    aer_backend = AerStateBackend()
    aer_handle = aer_backend.process_circuit(c)
    cu_handle = cu_backend.process_circuits([c])
    aer_result = aer_backend.get_result(aer_handle).get_state()
    cu_result = cu_backend.get_result(cu_handle[0]).get_state().array
    assert np.allclose(cu_result, aer_result)

def test_variety_of_gates() -> None:
    c = Circuit(3)
    c.H(0)
    c.S(1)
    c.T(2)
    c.CX(0, 1)
    c.Rz(0.4, 2)
    c.SX(0)
    c.CX(1, 2)
    c.Z(0)
    c.Y(1)
    c.X(2)
    cu_backend = CuStateVecStateBackend()
    aer_backend = AerStateBackend()
    aer_handle = aer_backend.process_circuit(c)
    cu_handle = cu_backend.process_circuits([c])
    aer_result = aer_backend.get_result(aer_handle).get_state()
    cu_result = cu_backend.get_result(cu_handle[0]).get_state().array
    assert np.allclose(cu_result, aer_result)
