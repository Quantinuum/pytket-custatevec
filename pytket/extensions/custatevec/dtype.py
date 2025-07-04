import numpy as np  # noqa: EXE002, D100
from cuquantum import cudaDataType
from numpy.typing import DTypeLike

_cuquantum_to_np_dtype_map = {
    cudaDataType.CUDA_R_16F : np.float16,
    cudaDataType.CUDA_R_32F : np.float32,
    cudaDataType.CUDA_R_64F : np.float64,
    cudaDataType.CUDA_C_32F : np.complex64,
    cudaDataType.CUDA_C_64F : np.complex128,
}

def cuquantum_to_np_dtype(cuquantum_dtype : cudaDataType) -> DTypeLike:
    """Convert a cuQuantum CUDA data type to the corresponding NumPy data type.

    Args:
        cuquantum_dtype (cudaDataType): The cuQuantum CUDA data type.

    Returns:
        DTypeLike: The corresponding NumPy data type.

    Raises:
        NotImplementedError: If the provided CUDA data type is not supported.
    """
    try:
        return _cuquantum_to_np_dtype_map[cuquantum_dtype]
    except:  # noqa: E722
        raise NotImplementedError(f"Cuda dtype {cudaDataType(cuquantum_dtype).name} not implemented in dtype.py; open an issue on GitHub.")  # noqa: B904, E501, EM102, RUF100, W292
