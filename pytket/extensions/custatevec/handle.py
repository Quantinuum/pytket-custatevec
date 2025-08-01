# Copyright Quantinuum  # noqa: D100
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
from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import Self

if TYPE_CHECKING:
    import types
    from logging import Logger

    from cupy.cuda import Stream


from .utils import INSTALL_CUDA_ERROR_MESSAGE

try:
    import cupy as cp
    import cuquantum.custatevec as cusv
except ImportError as _cuda_import_err:
    raise RuntimeError(INSTALL_CUDA_ERROR_MESSAGE.format(getattr(_cuda_import_err, "name", None))) from _cuda_import_err


class CuStateVecHandle:
    """Initialise the cuStateVec library with automatic workspace memory management.

    Note:
        Always use as ``with CuStateVecHandle() as libhandle:`` so that cuStateVec
        handles are automatically destroyed at the end of execution.

    Attributes:
        handle (int): The cuStateVec library handle created by this initialisation.
        device_id (int): The ID of the device (GPU) where cuStateVec is initialised.
            If not provided, defaults to ``cp.cuda.Device()``.
    """

    stream: Stream

    def __init__(self, device_id: int | None = None) -> None:
        """Initialise the cuStateVec library with automatic workspace memory management.

        Args:
            device_id (int | None): The ID of the device (GPU) to use. If None,
                defaults to the current device set by CuPy.
        """
        self._is_destroyed = False

        # Make sure CuPy uses the specified device
        dev = cp.cuda.Device(device_id)
        dev.use()

        self.dev = dev
        self.device_id = dev.id

        self._handle = cusv.create()  # type: ignore[no-untyped-call]

        def malloc(size: int, stream: Stream) -> int:
            return int(cp.cuda.runtime.mallocAsync(size, stream))

        def free(ptr: int, stream: Stream) -> None:
            cp.cuda.runtime.freeAsync(ptr, stream)

        handler = (malloc, free, "memory_handler")
        stream = cp.cuda.Stream()
        self.stream = stream
        cusv.set_device_mem_handler(self._handle, handler)  # type: ignore[no-untyped-call]
        cusv.set_stream(self._handle, stream.ptr)  # type: ignore[no-untyped-call]

    @property
    def handle(self) -> int:
        """Returns the cuStateVec library handle.

        Raises:
            RuntimeError: If the handle has been destroyed or is out of scope.
        """
        if self._is_destroyed:
            raise RuntimeError(
                "The cuStateVec library handle is out of scope.",
                "See the documentation of CuStateVecHandle.",
            )
        return int(self._handle)

    def destroy(self) -> None:
        """Destroys the memory handle, releasing memory.

        Only call this method if you are initialising a ``CuStateVecHandle`` outside
        a ``with CuStateVecHandle() as libhandle`` statement.
        """
        cusv.destroy(self._handle)  # type: ignore[no-untyped-call]
        self._is_destroyed = True

    def __enter__(self) -> Self:
        """Enter the runtime context and return the handle."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit the runtime context and destroy the handle.

        Args:
            exc_type (type[BaseException] | None): The exception type, if an exception occurred.
            exc_value (BaseException | None): The exception instance, if an exception occurred.
            exc_tb (types.TracebackType | None): The traceback object, if an exception occurred.
        """
        self.destroy()

    def print_device_properties(self, logger: Logger) -> None:
        """Prints local GPU properties."""
        device_props = cp.cuda.runtime.getDeviceProperties(self.dev.id)
        logger.info("===== device info ======")
        logger.info("GPU-name: %s", device_props["name"].decode())
        logger.info("GPU-clock: %s", device_props["clockRate"])
        logger.info("GPU-memoryClock: %s", device_props["memoryClockRate"])
        logger.info("GPU-nSM: %s", device_props["multiProcessorCount"])
        logger.info("GPU-major: %s", device_props["major"])
        logger.info("GPU-minor: %s", device_props["minor"])
        logger.info("========================")
