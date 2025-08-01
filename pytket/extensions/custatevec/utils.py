# Copyright Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module with miscellaneous utility functions and constants."""

from pytket._tket.circuit import Circuit
from pytket._tket.unit_id import Bit
from pytket.circuit import OpType, Qubit

INSTALL_CUDA_ERROR_MESSAGE = r"""
No installation of {} found!

`pytket-custatevec` is available for Python 3.10, 3.11 and 3.12 on Linux.
In order to use it, you need access to a Linux machine (or WSL) with an NVIDIA GPU of
Compute Capability +7.0 (check it https://developer.nvidia.com/cuda-gpus) and
have `cuda-toolkit` installed; this can be done with the command

sudo apt install cuda-toolkit

You need to install `cuquantum-python` before `pytket-custatevec`.
The recommended way to install these dependency is using conda:

conda install -c conda-forge cuquantum-python

This will automatically pull all other CUDA-related dependencies.

For more details, including how to install these dependencies via pip or how to manually specify the CUDA version,
read the install instructions in the official cuQuantum documentation https://docs.nvidia.com/cuda/cuquantum/latest/getting-started/index.html.
"""


def _remove_meas_and_implicit_swaps(circ: Circuit) -> tuple[Circuit, dict[Qubit, Bit]]:
    """Converts a pytket Circuit to an equivalent circuit without measurements or implicit swaps.

    Measurements are extracted and returned as a mapping between qubits and bits.
    This function only supports end-of-circuit measurements. Any mid-circuit
    measurements or operations on classical bits will raise an error.

    Args:
        circ (Circuit): The input pytket Circuit.

    Returns:
        tuple[Circuit, dict[Qubit, Bit]]:
            - A new Circuit object with measurements and implicit swaps removed.
            - A dictionary mapping measured Qubits to their corresponding Bits.

    Raises:
        ValueError: If the circuit contains mid-circuit measurements or operations
        on classical bits.
    """
    pure_circ = Circuit()
    for q in circ.qubits:
        pure_circ.add_qubit(q)
    q_perm = circ.implicit_qubit_permutation()

    measure_map = {}
    # Track measured Qubits to identify mid-circuit measurement
    measured_qubits = set()

    for command in circ:
        cmd_qubits = [q_perm[q] for q in command.qubits]

        for q in cmd_qubits:
            if q in measured_qubits:
                raise ValueError("Circuit contains a mid-circuit measurement")

        if command.op.type == OpType.Measure:
            measure_map[cmd_qubits[0]] = command.bits[0]
            measured_qubits.add(cmd_qubits[0])
        else:
            if command.bits:
                raise ValueError("Circuit contains an operation on a bit")
            pure_circ.add_gate(command.op, cmd_qubits)

    pure_circ.add_phase(circ.phase)
    return pure_circ, measure_map
