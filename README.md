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

Code style can be checked locally using [pre-commit](https://pre-commit.com/) hooks; run pre-commit 
before commiting your changes and opening a pull request by executing
```
pre-commit run
```
This will automatically:
* Format code using [ruff](https://pypi.org/project/ruff/) with default options.
* Do static type checking using [mypy](https://mypy.readthedocs.io/en/stable/).
* Lint using [ruff](https://pypi.org/project/ruff/) to check compliance
with a set of style requirements (listed in `ruff.toml`).

Compliance with the above checks is checked by continuous integration before a pull request
 can be merged. 

#### Docstrings

We use the Google style docstrings, please see this
[page](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for
reference.

### Tests

To run the tests for a module:

```shell
pip install pytket-custatevec[test]
pytest tests/
```

When adding a new feature, please add a test for it. When fixing a bug, please
add a test that demonstrates the fix.
