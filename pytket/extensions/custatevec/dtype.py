from pytket.extensions.custatevec.wrapper.cupy import cudaDataType

import numpy as np
from numpy.typing import DTypeLike

_cuquantum_to_np_dtype_map = {
    cudaDataType.CUDA_R_16F : np.float16,
    cudaDataType.CUDA_R_32F : np.float32,
    cudaDataType.CUDA_R_64F : np.float64,
    cudaDataType.CUDA_C_32F : np.complex64,
    cudaDataType.CUDA_C_64F : np.complex128,
}

def cuquantum_to_np_dtype(cuquantum_dtype : cudaDataType) -> DTypeLike:
    try:
        return _cuquantum_to_np_dtype_map[cuquantum_dtype]
    except:
        raise NotImplementedError(f"Cuda dtype {cudaDataType(cuquantum_dtype).name} not implemented in dtype.py; open an issue on GitHub.")