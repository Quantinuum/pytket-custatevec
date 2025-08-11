# pytket-custatevec

[![Slack](https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white)](https://tketusers.slack.com/join/shared_invite/zt-18qmsamj9-UqQFVdkRzxnXCcKtcarLRA#)
[![Stack Exchange](https://img.shields.io/badge/StackExchange-%23ffffff.svg?style=for-the-badge&logo=StackExchange)](https://quantumcomputing.stackexchange.com/tags/pytket)

[Pytket](https://docs.quantinuum.com/tket/api-docs/) is a python module for interfacing
with tket, a quantum computing toolkit and optimising compiler developed by Quantinuum.

[cuStateVec](https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/index.html) is a
high-performance library for statevector simulation, developed by NVIDIA.
It is part of the [cuQuantum](https://docs.nvidia.com/cuda/cuquantum/latest/index.html) SDK -
a high-performance library aimed at quantum circuit simulations on the NVIDIA GPUs.

`pytket-custatevec` is an extension to `pytket` that allows `pytket` circuits and
expectation values to be simulated using `cuStateVec` via an interface to
[cuQuantum Python](https://docs.nvidia.com/cuda/cuquantum/latest/python/index.html).

Some useful links:
- [API Documentation](https://docs.quantinuum.com/tket/extensions/pytket-custatevec/)

## Getting started

`pytket-custatevec` is available for Python 3.10, 3.11 and 3.12 on Linux.
In order to use it, you need access to a Linux machine (or WSL) with an NVIDIA GPU of
Compute Capability +7.0 (check it [here](https://developer.nvidia.com/cuda-gpus)) and
have `cuda-toolkit` installed; this can be done with the command

```shell
sudo apt install cuda-toolkit
```

You need to install `cuquantum-python` before `pytket-custatevec`.
The recommended way to install these dependency is using conda:

```shell
conda install -c conda-forge cuquantum-python
```
This will automatically pull all other CUDA-related dependencies.

For more details, including how to install these dependencies via pip or how to manually specify the CUDA version,
read the [install instructions in the official cuQuantum documentation](https://docs.nvidia.com/cuda/cuquantum/latest/getting-started/index.html).




## Bugs, support and feature requests

Please file bugs and feature requests on the Github
[issue tracker](https://github.com/CQCL/pytket-custatevec/issues).

## Development

To install an extension in editable mode, from its root folder run:

```shell
pip install -e .
```

## Contributing

Pull requests are welcome. To make a PR, first fork the repo, make your proposed
changes on the `main` branch, and open a PR from your fork. If it passes
tests and is accepted after review, it will be merged in.

### Code style

#### Docstrings

We use the Google style docstrings, please see this
[page](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for
reference.

#### Formatting

All code should be formatted using [ruff](https://pypi.org/project/ruff/), with default options.

#### Type annotation

On the CI, [pyright](https://mypy.readthedocs.io/en/stable/) is used as a static
type checker and all submissions must pass its checks. You should therefore run
`mypy` locally on any changed files before submitting a PR. Because of the way
extension modules embed themselves into the `pytket` namespace this is a little
complicated, but it should be sufficient to run the script `mypy-check`
and passing as a single argument the root directory of the module to test. The directory
path should end with a `/`. For example, to run mypy on all Python files in this
repository, when in the root folder, run:

```shell
./mypy-check ./
```
The script requires `mypy` 0.800 or above.

#### Linting

We use [ruff](https://pypi.org/project/ruff/) on the CI to check compliance
with a set of style requirements (listed in `ruff.toml`). You should run
`ruff` over any changed files before submitting a PR.

### Tests

To run the tests for a module:

```shell
pip install pytket-custatevec[test]
pytest tests/
```

When adding a new feature, please add a test for it. When fixing a bug, please
add a test that demonstrates the fix.
