"""Tests for package metadata consistency."""

import ast
from pathlib import Path

import tomllib


def test_extension_metadata_version_matches_project_version() -> None:
    """Check backend-reported extension metadata matches package metadata."""
    root = Path(__file__).parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    metadata_tree = ast.parse((root / "pytket/extensions/custatevec/_metadata.py").read_text())

    metadata = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in metadata_tree.body
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
    }

    assert metadata["__extension_version__"] == project["project"]["version"]
