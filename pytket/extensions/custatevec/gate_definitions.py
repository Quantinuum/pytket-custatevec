import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from cuquantum.bindings._utils import cudaDataType
from numpy.typing import DTypeLike, NDArray

try:
    import cupy as cp  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import cupy", ImportWarning)

from pytket.extensions.custatevec.gate_classes import (
    CuStateVecMatrix,
    ParameterizedGate,
    UnparameterizedGate,
)

from .dtype import cuquantum_to_np_dtype

_I = np.eye(2)

I = UnparameterizedGate("I", _I)

_0 = np.zeros((2, 2))

_X = np.array([[0.0, 1.0], [1.0, 0.0]])
X = UnparameterizedGate("X", _X)

_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
Y = UnparameterizedGate("Y", _Y)

_Z = np.array([[1.0, 0.0], [0.0, -1.0]])
Z = UnparameterizedGate("Z", _Z)

_H = np.array([[1.0, 1.0], [1.0, -1]]) / np.sqrt(2)
H = UnparameterizedGate("H", _H)

_S = np.array([[1.0, 0.0], [0.0, 1.0j]])
S = UnparameterizedGate("S", _S)

T = UnparameterizedGate("T", np.array([[1.0, 0.0], [0.0, np.exp(np.pi * 1.0j / 4)]]))

_V = np.array([[1.0, -1.0j], [-1.0j, 1.0]]) / np.sqrt(2)
V = UnparameterizedGate("V", _V)

_Vdg = np.array([[1.0, 1.0j], [1.0j, 1.0]]) / np.sqrt(2)

_SX = np.array([[1.0 + 1.0j, 1.0 - 1.0j], [1.0 - 1.0j, 1.0 + 1.0j]]) / 2
SX = UnparameterizedGate("SX", _SX)

ECR = UnparameterizedGate("ECR", np.block([[_0, _Vdg], [_V, _0]]))

_SWAP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
)

SWAP = UnparameterizedGate("SWAP", _SWAP) # invariant under qubit permutation

def _Rx(params: Sequence[float], dtype: DTypeLike) -> NDArray[Any]:
    param_pi_2 = params[0] * np.pi / 2
    return (
        np.cos(param_pi_2, dtype=dtype) * _I
        - np.sin(param_pi_2, dtype=dtype) * _X * 1.0j
    )


Rx = ParameterizedGate("Rx", _Rx, 1, 1)


def _Ry(params: Sequence[float], dtype: DTypeLike) -> NDArray[Any]:
    param_pi_2 = params[0] * np.pi / 2
    return (
        np.cos(param_pi_2, dtype=dtype) * _I
        - np.sin(param_pi_2, dtype=dtype) * _Y * 1.0j
    )


Ry = ParameterizedGate("Ry", _Ry, 1, 1)


def _Rz(params: Sequence[float], dtype: DTypeLike) -> NDArray[Any]:
    param_pi_2 = params[0] * np.pi / 2
    return (
        np.cos(param_pi_2, dtype=dtype) * _I
        - np.sin(param_pi_2, dtype=dtype) * _Z * 1.0j
    )


Rz = ParameterizedGate("Rz", _Rz, 1, 1)

def _TK1(params: Sequence[float], dtype: DTypeLike) -> NDArray[Any]:
    return (
        _Rz([params[0]], dtype=dtype)
        @ _Rx([params[1]], dtype=dtype)
        @ _Rz([params[2]], dtype=dtype)
    )


TK1 = ParameterizedGate("TK1", _TK1, 1, 3)

def _U3(params: Sequence[float], dtype: DTypeLike) -> NDArray[Any]:
    return (
        np.exp((params[1] + params[2]) * np.pi * 1.0j / 2, dtype=dtype)
        * _Rz([params[1]], dtype=dtype)
        @ _Ry([params[0]], dtype=dtype)
        @ _Rz([params[2]], dtype=dtype)
    )


def _U1(params: Sequence[float], dtype: DTypeLike) -> NDArray[Any]:
    return _U3([0, 0, params[0]], dtype=dtype)


def _U2(params: Sequence[float], dtype: DTypeLike) -> NDArray[Any]:
    return _U3([0.5, params[0], params[1]], dtype=dtype)


U1 = ParameterizedGate("U1", _U1, 1, 1)
U2 = ParameterizedGate("U2", _U2, 1, 2)
U3 = ParameterizedGate("U3", _U3, 1, 3)


def _ISWAP(params: Sequence[float], dtype: DTypeLike) -> NDArray[Any]:
    param_pi_2 = params[0] * np.pi / 2
    c = np.cos(param_pi_2, dtype=dtype)
    i_s = 1.0j * np.sin(param_pi_2, dtype=dtype)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, i_s, 0.0],
            [0.0, i_s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )


ISWAP = ParameterizedGate("ISWAP", _ISWAP, 2, 1)


def _PhasedISWAP(params: Sequence[float], dtype: DTypeLike) -> NDArray[Any]:
    param1_pi_2 = params[1] * np.pi / 2
    c = np.cos(param1_pi_2, dtype=dtype)
    i_s = 1.0j * np.sin(param1_pi_2, dtype=dtype)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, i_s * np.exp(2.0j * np.pi * params[0], dtype=dtype), 0.0],
            [0.0, i_s * np.exp(-2.0j * np.pi * params[0], dtype=dtype), c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )


PhasedISWAP = ParameterizedGate("PhasedISWAP", _PhasedISWAP, 2, 2)


def _XXPhase(params: Sequence[float], dtype: DTypeLike) -> NDArray[Any]:
    param_pi_2 = params[0] * np.pi / 2
    c = np.cos(param_pi_2, dtype=dtype)
    i_s = 1.0j * np.sin(param_pi_2, dtype=dtype)
    return np.array(
        [
            [c, 0.0, 0.0, -i_s],
            [0.0, c, -i_s, 0.0],
            [0.0, -i_s, c, 0.0],
            [-i_s, 0.0, 0.0, c],
        ],
    ).reshape((2,) * 4)


XXPhase = ParameterizedGate("XXPhase", _XXPhase, 2, 1)


def _YYPhase(params: Sequence[float], dtype: DTypeLike) -> NDArray[Any]:
    param_pi_2 = params[0] * np.pi / 2
    c = np.cos(param_pi_2, dtype=dtype)
    i_s = 1.0j * np.sin(param_pi_2, dtype=dtype)
    return np.array(
        [
            [c, 0.0, 0.0, i_s],
            [0.0, c, -i_s, 0.0],
            [0.0, -i_s, c, 0.0],
            [i_s, 0.0, 0.0, c],
        ],
    )


YYPhase = ParameterizedGate("YYPhase", _YYPhase, 2, 1)


def _ZZPhase(params: Sequence[float], dtype: DTypeLike) -> NDArray[Any]:
    param_pi_2 = params[0] * np.pi / 2
    e_m = np.exp(-1.0j * param_pi_2, dtype=dtype)
    e_p = np.exp(1.0j * param_pi_2, dtype=dtype)
    return np.diag(np.array([e_m, e_p, e_p, e_m]))


ZZPhase = ParameterizedGate("ZZPhase", _ZZPhase, 2, 1)

ZZMax = UnparameterizedGate("ZZMax", _ZZPhase([0.5], None))


def _PhasedX(params: Sequence[float], dtype: DTypeLike) -> NDArray[Any]:
    return (
        _Rz([params[1]], dtype=dtype)
        @ _Rx([params[0]], dtype=dtype)
        @ _Rz([-params[1]], dtype=dtype)
    )


PhasedX = ParameterizedGate("PhasedX", _PhasedX, 1, 2)

gate_list = [
    X,
    Y,
    Z,
    H,
    S,
    T,
    V,
    SX,
    ECR,
    SWAP,
    Rx,
    Ry,
    Rz,
    TK1,
    U1,
    U2,
    U3,
    ISWAP,
    PhasedISWAP,
    XXPhase,
    YYPhase,
    ZZPhase,
    ZZMax,
    PhasedX,
]
gate_dict = {g.name: g for g in gate_list}

_control_to_gate_map: dict[str, tuple[str, int]] = {
    "CX": ("X", 1),
    "CY": ("Y", 1),
    "CZ": ("Z", 1),
    "CH": ("H", 1),
    "CV": ("V", 1),
    "CSX": ("SX", 1),
    "CCX": ("X", 2),
    "CSWAP": ("CSWAP", 1),
    "CRx": ("Rx", 1),
    "CRy": ("Ry", 1),
    "CU1": ("U1", 1),
    "CU2": ("U2", 1),
    "CU3": ("U3", 1),
}


def get_uncontrolled_gate(name: str) -> tuple[str, int]:
    """Retrieve the corresponding uncontrolled gate name and control level.

    Args:
        name (str): The name of the controlled gate.

    Returns:
        tuple[str, int]: A tuple containing the uncontrolled gate name and the control level.
    """
    try:
        return _control_to_gate_map[name]
    except KeyError:
        return (name, 0)


def get_gate_matrix(
    gate_name: str, parameters: Sequence[float], cuda_dtype: cudaDataType,
) -> CuStateVecMatrix:
    """Retrieve the matrix representation of a quantum gate.

    Args:
        gate_name (str): The name of the gate.
        parameters (Sequence[float]): The parameters for the gate.
        cuda_dtype (cudaDataType): The CUDA data type for the gate matrix.

    Returns:
        CuStateVecMatrix: The matrix representation of the gate in CuPy format.

    Raises:
        ValueError: If the gate name is not found in the gate dictionary.
    """
    dtype = cuquantum_to_np_dtype(cuda_dtype)
    try:
        gate = gate_dict[gate_name]
        return CuStateVecMatrix(
            cp.array(gate.get(parameters, dtype), dtype=dtype), cuda_dtype,
        )
    except KeyError:
        raise ValueError(f"Gate {gate_name} not found")
