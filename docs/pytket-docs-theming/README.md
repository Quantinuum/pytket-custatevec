# pytket-docs-theming

Docs configuration for docs.quantinuum.com/tket based on [Quantinuum-sphinx](https://github.com/CQCL/quantinuum-sphinx). 

This repository is intended to be used as a git submodule in documentation source repositories. 

```shell
git submodule add -b main https://github.com/CQCL/pytket-docs-theming.git
```

## Contents

Contains a single source of truth for the following docs config

1. Sphinx configuration - ([conf.py](https://github.com/CQCL/pytket-docs-theming/blob/main/conf.py))
2. Navbar configuration (CURRENTLY NOT USED!) - ([nav-config.js](https://github.com/CQCL/pytket-docs-theming/blob/main/_static/nav-config.js))
3. [Quantinuum-sphinx submodule version](https://github.com/CQCL/quantinuum-sphinx/tree/33287684e58d2b38ab5867c8e79bcd43460676df)
4. Python packages required to build the docs for the pytket extensions modules. These are specified in [pyproject.toml](https://github.com/CQCL/pytket-docs-theming/blob/main/extensions/pyproject.toml) and [poetry.lock](https://github.com/CQCL/pytket-docs-theming/blob/main/extensions/poetry.lock) (see the [README](https://github.com/CQCL/pytket-docs-theming/tree/main/extensions#building-the-api-docs-for-pytket-extensions) for instructions).
