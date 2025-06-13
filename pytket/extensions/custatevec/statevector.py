
import cupy as cp  # type: ignore

from .gate_classes import *

import cuquantum  # type: ignore
from cuquantum import cudaDataType


class CuStateVector:
    array: cp.ndarray
    cuda_dtype: cudaDataType
    n_qubits: int

    def __init__(self, array: cp.ndarray, cuda_dtype: cudaDataType) -> None:
        self.array = array
        _n_qubits = math.log2(array.size)
        if not _n_qubits.is_integer():
            raise ValueError()
        self.n_qubits = int(_n_qubits)
        self.cuda_dtype = cuda_dtype

    def apply_phase(self, phase : float) -> None:
        self.array *= phase