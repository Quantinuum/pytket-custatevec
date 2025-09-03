import math  # noqa: D100

from .utils import INSTALL_CUDA_ERROR_MESSAGE

try:
    import cupy as cp
    from cuquantum import cudaDataType
except ImportError as _cuda_import_err:
    raise RuntimeError(INSTALL_CUDA_ERROR_MESSAGE.format(getattr(_cuda_import_err, "name", None))) from _cuda_import_err

from .gate_classes import *  # noqa: F403


class CuStateVector:
    """Represents a state vector in cuStateVec.

    This class provides methods to manipulate and apply operations
    on a state vector.
    """

    n_qubits: int
    shape: tuple[int, ...]

    def __init__(self, array: cp.ndarray, cuda_dtype: cudaDataType) -> None:
        """Initializes the statevector with the given array and CUDA data type.

        Args:
            array (cp.ndarray): The statevector array represented as a CuPy ndarray.
                The size of the array must be a power of 2.
            cuda_dtype (cudaDataType): The CUDA data type of the statevector.

        Raises:
            ValueError: If the size of the array is not a power of 2.

        Attributes:
            array (cp.ndarray): The input statevector array.
            n_qubits (int): The number of qubits, calculated as log2 of the array size.
            cuda_dtype (cudaDataType): The CUDA data type of the statevector.
            shape (tuple): The shape of the input array.
        """
        self.array = array
        _n_qubits = math.log2(array.size)
        if not _n_qubits.is_integer():
            raise ValueError
        self.n_qubits = int(_n_qubits)
        self.cuda_dtype = cuda_dtype
        self.shape = array.shape

    def apply_phase(self, phase: float) -> None:
        """Apply a global phase to the state vector.

        Args:
            phase (float): The phase shift to apply, in units of pi.
        """
        self.array *= cp.exp(1j * cp.pi * phase)
