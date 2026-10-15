"""Sphinx configuration for the imported CPU and optional CUDA package."""

import importlib.metadata
import pathlib
import tomllib

import cgpy

project = "Conjugate Gradient Method Library"
author = "Conjugate Gradient Method Library contributors"
project_root = pathlib.Path(__file__).resolve().parents[1]
project_metadata = tomllib.loads((project_root / "pyproject.toml").read_text())
release = project_metadata["project"].get(
    "version"
) or importlib.metadata.version("cgpy")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
]
source_suffix = {".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_build"]
html_theme = "alabaster"
html_title = f"cgpy {release}"
autodoc_member_order = "bysource"
autodoc_typehints = "none"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False
napoleon_use_ivar = True
doctest_global_setup = "import numpy as np\nimport cgpy\nfrom cgpy import cuda"

# Import failures must stop the build. Make selects the staged package in
# its build directory; installed-wheel builds need no path override.
cgpy.build_info()
