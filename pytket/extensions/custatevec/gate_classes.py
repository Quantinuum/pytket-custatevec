import abc  # noqa: D100
import math
from collections.abc import Callable, Sequence
from typing import Any

from .utils import INSTALL_CUDA_ERROR_MESSAGE

try:
    import cupy as cp
    from cuquantum import cudaDataType
except ImportError as _cuda_import_err:
    raise RuntimeError(INSTALL_CUDA_ERROR_MESSAGE.format(getattr(_cuda_import_err, "name", None))) from _cuda_import_err


from numpy.typing import DTypeLike, NDArray


class Gate:
    """Abstract base class for quantum gates."""

    name: str

    @abc.abstractmethod
    def get(self, parameters: Sequence[float], dtype: DTypeLike) -> NDArray[Any]:
        """Return the matrix of the gate.

        Args:
            parameters (Sequence[float]): Parameters for the gate.
            dtype (DTypeLike): The desired data type of the output matrix.

        Returns:
            NDArray[Any]: The matrix of the gate, cast to the specified dtype.
        """

    @property
    @abc.abstractmethod
    def qubits(self) -> int:
        """Return the number of qubits that this gate acts on.

        Returns:
            int: The number of qubits.
        """

    @property
    def n_parameters(self) -> int:
        """Return the number of parameters for this gate.

        Returns:
            int: The number of parameters.
        """
        return 0


class UnparameterizedGate(Gate):
    """Represents a quantum gate with a fixed matrix and no parameters."""

    _matrix: NDArray[Any]
    _qubits: int

    def __init__(self, name: str, matrix: NDArray[Any]) -> None:
        """Initialize an UnparameterizedGate with a name and a fixed matrix.

        Args:
            name (str): The name of the gate.
            matrix (NDArray[Any]): The matrix representing the gate.
                Must have shape (2**q, 2**q) for some integer q.

        Raises:
            ValueError: If the matrix does not have a shape of (2**q, 2**q).
        """
        assert matrix.ndim == 2  # noqa: PLR2004, S101
        assert matrix.shape[0] == matrix.shape[1]  # noqa: S101

        d = matrix.shape[0]
        _q = math.log2(d)
        if _q.is_integer():
            q = int(_q)
        else:
            raise ValueError(
                "Matrix passed to UnparameterizedGate does not have shape (2**q, 2**q)",
            )
        self._qubits = q

        self.name = name
        self._matrix = matrix

    def get(self, parameters: Sequence[float], dtype: DTypeLike) -> NDArray[Any]:
        """Return the matrix of the gate.

        Args:
            parameters (Sequence[float]): Parameters for the gate. Should be empty
                for an unparameterized gate.
            dtype (DTypeLike): The desired data type of the output matrix.

        Returns:
            NDArray[Any]: The matrix of the gate, cast to the specified dtype.

        Raises:
            ValueError: If parameters are passed to an unparameterized gate.
        """
        if len(parameters) > 0:
            raise ValueError(f"Passed {len(parameters)} to an unparmeterised gate")
        return self._matrix.astype(dtype)

    @property
    def qubits(self) -> int:
        """Return the number of qubits that this gate acts on."""
        return self._qubits

    @property
    def n_parameters(self) -> int:
        """Return the number of parameters for this gate."""
        return 0

    @property
    def matrix(self) -> NDArray[Any]:
        """Return the matrix of the gate."""
        return self._matrix


class ParameterizedGate(Gate):
    """Represents a quantum gate with a parameterized function."""

    function: Callable[[Sequence[float], DTypeLike], NDArray[Any]]
    _n_parameters: int

    def __init__(
        self,
        name: str,
        function: Callable[[Sequence[float], DTypeLike], NDArray[Any]],
        qubits: int,
        n_parameters: int,
    ) -> None:
        """Initialize a ParameterizedGate with a name, function, number of qubits, and parameters.

        Args:
            name (str): The name of the gate.
            function (Callable[[Sequence[float], DTypeLike], NDArray[Any]]):
                A function that takes parameters and dtype, and returns a matrix.
            qubits (int): The number of qubits the gate acts on.
            n_parameters (int): The number of parameters for this gate.

        Raises:
            ValueError: If the number of qubits is less than 1 or if n_parameters is negative.
        """
        self.name = name
        self.function = function
        self._qubits = qubits
        self._n_parameters = n_parameters

    def get(self, parameters: Sequence[float], dtype: DTypeLike) -> NDArray[Any]:
        """Return the matrix of the gate based on the provided parameters.

        Args:
            parameters (Sequence[float]): Parameters for the gate.
            dtype (DTypeLike): The desired data type of the output matrix.

        Returns:
            NDArray[Any]: The matrix of the gate, computed by the function and cast to the specified dtype.
        """
        return self.function(parameters, dtype)

    @property
    def qubits(self) -> int:
        """Return the number of qubits the gate acts on.

        Returns:
            int: The number of qubits.
        """
        return self._qubits

    @property
    def n_parameters(self) -> int:
        """Return the number of parameters for this gate.

        Returns:
            int: The number of parameters.
        """
        return self._n_parameters


class CuStateVecMatrix:
    """Represents a matrix used in cuStateVec operations."""

    matrix: cp.ndarray
    cuda_dtype: cudaDataType
    n_qubits: int

    def __init__(self, matrix: cp.ndarray, cuda_dtype: cudaDataType) -> None:
        """Initialize a CuStateVecMatrix with a matrix and CUDA data type.

        Args:
            matrix (cp.ndarray): The matrix represented as a CuPy ndarray.
                Must have shape (2**q, 2**q) for some integer q.
            cuda_dtype (cudaDataType): The CUDA data type of the matrix.

        Raises:
            ValueError: If the matrix does not have a shape of (2**q, 2**q).
        """
        self.matrix = matrix
        self.cuda_dtype = cuda_dtype

        d = matrix.shape[0]
        _q = math.log2(d)
        if _q.is_integer():
            q = int(_q)
        else:
            raise ValueError(
                "Matrix passed to UnparameterizedGate does not have shape (2**q, 2**q)",
            )
        self.n_qubits = q
