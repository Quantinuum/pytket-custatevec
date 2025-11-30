## Development

To install an extension in editable mode, from its root folder run:

```shell
pip install -e .
```

### Contributing

Pull requests are welcome. To make a PR, first fork the repo, make your proposed changes on the `main` branch, and open a PR from your fork. If it passes tests and is accepted after review, it will be merged in.

### Code style

Code style can be checked locally using [pre-commit](https://pre-commit.com/) hooks; run pre-commit before committing your changes and opening a pull request by executing:

```shell
pre-commit run
```

This will automatically:

* Format code using [ruff](https://pypi.org/project/ruff/) with default options.
* Do static type checking using [mypy](https://mypy.readthedocs.io/en/stable/).
* Lint using [ruff](https://pypi.org/project/ruff/) to check compliance with a set of style requirements (listed in `ruff.toml`).

Compliance with the above checks is checked by continuous integration before a pull request can be merged.

#### Docstrings

We use the Google style docstrings, please see this [page](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for reference.

### Tests

To run the tests for a module:

```shell
pip install "pytket-custatevec[test]"
pytest tests/
```

When adding a new feature, please add a test for it. When fixing a bug, please add a test that demonstrates the fix.