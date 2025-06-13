import warnings
import math
import abc
from typing import Callable, Sequence, Any

import cuquantum
from cuquantum import cudaDataType

try:
    import cupy as cp  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import cupy", ImportWarning)

import numpy as np
from numpy.typing import DTypeLike, NDArray


class Gate:
    name: str

    @abc.abstractmethod
    def get(self, parameters: Sequence[float], dtype: DTypeLike) -> NDArray[Any]:
        pass

    @property
    @abc.abstractmethod
    def qubits(self) -> int:
        pass

    @property
    def n_parameters(self) -> int:
        return 0


class UnparameterizedGate(Gate):
    _matrix: NDArray[Any]
    _qubits: int

    def __init__(self, name: str, matrix: NDArray[Any]) -> None:
        assert matrix.ndim == 2
        assert matrix.shape[0] == matrix.shape[1]

        d = matrix.shape[0]
        _q = math.log2(d)
        if _q.is_integer():
            q = int(_q)
        else:
            raise ValueError(
                "Matrix passed to UnparameterizedGate does not have shape (2**q, 2**q)"
            )
        self._qubits = q

        self.name = name
        self._matrix = matrix

    def get(self, *parameters: float, dtype: DTypeLike) -> NDArray[Any]:
        if len(parameters) > 0:
            raise ValueError(f"Passed {len(parameters)} to an unparmeterised gate")
        return self._matrix.astype(dtype)

    @property
    def qubits(self) -> int:
        return self._qubits

    @property
    def n_parameters(self) -> int:
        return 0

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix


class ParameterizedGate(Gate):
    function: Callable[[Sequence[float], DTypeLike], NDArray[Any]]
    _n_parameters: int

    def __init__(
        self,
        name: str,
        function: Callable[[Sequence[float], DTypeLike], NDArray[Any]],
        qubits: int,
        n_parameters: int,
    ) -> None:
        self.name = name
        self.function = function
        self._qubits = qubits
        self._n_parameters = n_parameters

    def get(self, parameters: Sequence[float], dtype: DTypeLike) -> NDArray[Any]:
        return self.function(parameters, dtype)

    @property
    def qubits(self) -> int:
        return self._qubits

    @property
    def n_parameters(self) -> int:
        return self._n_parameters


class CuStateVecMatrix:
    matrix: cp.ndarray
    cuda_dtype: cudaDataType
    qubits: int

    def __init__(self, matrix: cp.ndarray, cuda_dtype: cudaDataType):
        self.matrix = matrix
        self.cuda_dtype = cuda_dtype

        d = matrix.shape[0]
        _q = math.log2(d)
        if _q.is_integer():
            q = int(_q)
        else:
            raise ValueError(
                "Matrix passed to UnparameterizedGate does not have shape (2**q, 2**q)"
            )
        self.qubits = q
