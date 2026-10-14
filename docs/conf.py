"""Sphinx configuration for the imported native CPU package."""

from importlib.metadata import version as distribution_version
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10, pinned in requirements/dev.lock.
    import tomli as tomllib

import cgpy

project = "Conjugate Gradient Method Library"
author = "Conjugate Gradient Method Library contributors"
project_root = Path(__file__).resolve().parents[1]
project_metadata = tomllib.loads((project_root / "pyproject.toml").read_text())
release = project_metadata["project"].get("version") or distribution_version("cgpy")
version = release

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.doctest"]
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
doctest_global_setup = "import numpy as np\nimport cgpy"

# Import failures must stop the build. Make supplies the chosen native build
# and source package through PYTHONPATH; installed-wheel builds need no override.
cgpy.build_info()
