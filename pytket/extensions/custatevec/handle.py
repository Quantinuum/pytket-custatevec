# Copyright Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
##
#     http://www.apache.org/licenses/LICENSE-2.0
##
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations  # type: ignore
import warnings
from logging import Logger

from typing import Any, Optional

try:
    import cupy as cp  # type: ignore
    from cupy.cuda import Stream
except ImportError:
    warnings.warn("failed to import cupy", ImportWarning)
try:
    import cuquantum
    import cuquantum.custatevec as cusv  # type: ignore
except ImportError:
    warnings.warn("failed to import cuquantum", ImportWarning)

from pytket.circuit import Circuit

from .statevector import CuStateVector


class CuStateVecHandle:
    """Initialise the cuStateVec library with automatic workspace memory
    management.

    Note:
        Always use as ``with CuStateVecHandle() as libhandle:`` so that cuStateVec
        handles are automatically destroyed at the end of execution.

    Attributes:
        handle (int): The cuStateVec library handle created by this initialisation.
        device_id (int): The ID of the device (GPU) where cuStateVec is initialised.
            If not provided, defaults to ``cp.cuda.Device()``.
    """

    stream: Stream

    def __init__(self, device_id: Optional[int] = None):
        self._is_destroyed = False

        # Make sure CuPy uses the specified device
        dev = cp.cuda.Device(device_id)
        dev.use()

        self.dev = dev
        self.device_id = dev.id

        self._handle = cusv.create()

        def malloc(size, stream):
            return cp.cuda.runtime.mallocAsync(size, stream)

        def free(ptr, size, stream):
            cp.cuda.runtime.freeAsync(ptr, stream)

        handler = (malloc, free, "memory_handler")
        stream = cp.cuda.Stream()
        self.stream = stream
        cusv.set_device_mem_handler(self._handle, handler)
        cusv.set_stream(self._handle, stream.ptr)

    @property
    def handle(self) -> Any:
        if self._is_destroyed:
            raise RuntimeError(
                "The cuStateVec library handle is out of scope.",
                "See the documentation of CuStateVecHandle.",
            )
        return self._handle

    def destroy(self) -> None:
        """Destroys the memory handle, releasing memory.

        Only call this method if you are initialising a ``CuStateVecHandle`` outside
        a ``with CuStateVecHandle() as libhandle`` statement.
        """
        cusv.destroy(self._handle)
        self._is_destroyed = True

    def __enter__(self) -> CuStateVecHandle:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        self.destroy()

    def print_device_properties(self, logger: Logger) -> None:
        """Prints local GPU properties."""
        device_props = cp.cuda.runtime.getDeviceProperties(self.dev.id)
        logger.info("===== device info ======")
        logger.info("GPU-name: " + device_props["name"].decode())
        logger.info("GPU-clock: " + str(device_props["clockRate"]))
        logger.info("GPU-memoryClock: " + str(device_props["memoryClockRate"]))
        logger.info("GPU-nSM: " + str(device_props["multiProcessorCount"]))
        logger.info("GPU-major: " + str(device_props["major"]))
        logger.info("GPU-minor: " + str(device_props["minor"]))
        logger.info("========================")
