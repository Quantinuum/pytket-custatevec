<div align="center">
  <img src="docs/assets/logo.svg" height="100" alt="pytket-custatevec logo" />
  <h1>pytket-custatevec</h1>
  <p><strong>GPU-accelerated statevector simulation for pytket.</strong></p>

  <p>
    <a href="https://github.com/CQCL/pytket-custatevec/actions">
      <img src="https://img.shields.io/badge/build-passing-brightgreen?style=flat&logo=github" alt="Build" />
    </a>
    <a href="https://pypi.org/project/pytket-custatevec/">
      <img src="https://img.shields.io/badge/pypi-v0.0.1-blue?style=flat&logo=pypi" alt="PyPI" />
    </a>
    <a href="https://github.com/CQCL/pytket-custatevec/blob/main/LICENSE">
      <img src="https://img.shields.io/badge/license-Apache%202.0-green?style=flat" alt="License" />
    </a>
    <a href="https://tketusers.slack.com/">
      <img src="https://img.shields.io/badge/Slack-TKET%20Users-4A154B?logo=slack" alt="Slack" />
    </a>
  </p>

  <h3>
    <a href="https://cqcl.github.io/pytket-custatevec/">📚 Read the Full Documentation</a>
  </h3>
</div>

---

**pytket-custatevec** is a high-performance extension for [pytket](https://tket.quantinuum.com/) that allows quantum circuits to be simulated on NVIDIA GPUs.

It acts as a bridge to the [NVIDIA cuQuantum](https://developer.nvidia.com/cuquantum-sdk) SDK, utilizing `cuStateVec` to optimize memory usage and gate execution speed for large statevector simulations.

## 🚀 Features

* **High Performance:** Significantly faster than CPU-based simulators for large qubit counts (20-30+).
* **Seamless Integration:** Works as a standard `pytket` Backend.
* **Optimized Memory:** Manages GPU VRAM efficiently for complex simulations.

## 🏛 Architecture

```mermaid
flowchart LR
    %% Node Definitions
    User([User Code])
    Backend[CuStateVecBackend]
    SDK[NVIDIA cuStateVec]
    GPU[NVIDIA GPU]

    %% Connections
    User --> |pytket Circuit| Backend
    Backend --> |cuQuantum Python| SDK
    SDK -.-> |CUDA| GPU

    %% Styles for GitHub Compatibility (Standard Colors)
    classDef default stroke-width:2px;
    
    %% User: Light Grey fill, Dark text
    classDef user fill:#F5F5F5,stroke:#333333,color:#333;

    %% Backend: Light Teal fill, Dark Teal text
    classDef pytket fill:#E0F2F1,stroke:#00796b,color:#004D40;
    
    %% NVIDIA SDK: Bright Green, Dark text
    classDef nvidia fill:#76b900,stroke:#558600,color:#000;
    
    %% Hardware: Black fill, White text, Dashed border
    classDef hardware fill:#121212,stroke:#E0E0E0,color:#fff,stroke-dasharray: 5 5;

    %% Apply Styles
    class User user;
    class Backend pytket;
    class SDK nvidia;
    class GPU hardware;
```

## 📦 Installation

**1. Prerequisites**
You need access to a machine with an NVIDIA GPU (Compute Capability 7.0+) and `cuda-toolkit` installed.

```shell
sudo apt install cuda-toolkit
```

**2. Install Dependencies (Conda recommended)**
```shell
conda install -c conda-forge cuquantum-python
```

**3. Install Package**
```shell
pip install pytket-custatevec
```

## ⚡ Quick Start

```python
from pytket import Circuit
from pytket.extensions.custatevec import CuStateVecStateBackend

# 1. Define a circuit
circ = Circuit(2).H(0).CX(0, 1)

# 2. Initialize the GPU backend
backend = CuStateVecStateBackend()
compiled_circ = backend.get_compiled_circuit(circ)

# 3. Run on GPU
statevector = backend.run_circuit(compiled_circ).get_state()
print(statevector)
```

## 💻 Development

### Setup
To install the extension in editable mode for development:

```shell
pip install -e ".[dev,docs,test]"
```

### Code Style
We use `pre-commit` to maintain code quality. Before committing, run:

```shell
pre-commit run
```
This handles formatting (ruff), type checking (mypy), and linting.

### Testing
To run the test suite (requires a GPU environment):

```shell
pytest tests/
```

## 📄 Citing

If you use `pytket-custatevec` in your research, please cite it as follows:

```bibtex
@software{pytket_custatevec,
  author = {Quantinuum},
  title = {pytket-custatevec: GPU-accelerated statevector simulation},
  url = {[https://github.com/CQCL/pytket-custatevec](https://github.com/CQCL/pytket-custatevec)},
  year = {2025}
}
```