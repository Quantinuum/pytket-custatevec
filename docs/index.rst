pytket-custatevec
==================

``pytket-custatevec`` is an extension to ``pytket`` that allows ``pytket`` circuits and
expectation values to be simulated using `cuStateVec <https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/index.html>`_.

`cuStateVec <https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/index.html>`_ is a
high-performance library for statevector computations, developed by NVIDIA.
It is part of the `cuQuantum <https://docs.nvidia.com/cuda/cuquantum/latest/index.html>`_ SDK --
a high-performance library aimed at quantum circuit simulations on the NVIDIA GPU chips.

``pytket-custatevec`` is available for Python 3.10, 3.11 and 3.12 on Linux.
In order to use it, you need access to a Linux machine (or WSL) with an NVIDIA GPU of
Compute Capability +7.0 (check it `here <https://developer.nvidia.com/cuda-gpus>`_).
You will need to install ``cuda-toolkit`` and ``cuquantum-python`` before ``pytket-custatevec``;
for instance, in Ubuntu 24.04:

::

   sudo apt install cuda-toolkit
   pip install cuquantum-python
   pip install pytket-custatevec

Alternatively, you may install cuQuantum Python following their `instructions <https://docs.nvidia.com/cuda/cuquantum/latest/getting-started/index.html>`_
using ``conda-forge``. This will include the necessary dependencies from CUDA toolkit. Then, you may install
``pytket-custatevec`` using ``pip``.

.. toctree::
   :caption: Changelog

   changelog.rst

.. toctree::
   :caption: Useful links

   Issue tracker <https://github.com/CQCL/pytket-custatevec/issues>
   PyPi <https://pypi.org/project/pytket-custatevec/>
