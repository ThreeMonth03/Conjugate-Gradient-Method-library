"""Check public typing from an isolated installed or staged package consumer."""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        type=pathlib.Path,
        default=pathlib.Path(sys.executable),
        help="Interpreter containing the installed package and NumPy.",
    )
    parser.add_argument(
        "--package-root",
        type=pathlib.Path,
        help="Built Python staging directory; omitted for installed wheels.",
    )
    options = parser.parse_args()
    fixtures = pathlib.Path(__file__).resolve().parents[1] / "tests/typing"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("MYPYPATH", None)
    with tempfile.TemporaryDirectory(prefix="cgpy-typing-") as directory:
        consumer = pathlib.Path(directory)
        for name in ("valid.py", "invalid.py"):
            shutil.copy2(fixtures / name, consumer / name)
        if options.package_root is not None:
            shutil.copytree(
                options.package_root.resolve() / "cgpy",
                consumer / "cgpy",
                ignore=shutil.ignore_patterns("__pycache__"),
            )
        # Exercise imports separately; static fixtures never allocate or solve.
        probe = """
import inspect
import pathlib
import sys

import cgpy
from cgpy import cuda

package = pathlib.Path(cgpy.__file__).resolve().parent
expected = pathlib.Path(sys.argv[1]).resolve()
assert package.is_relative_to(expected), (package, expected)
for name in ('py.typed', '_core.pyi', '_cuda.pyi'):
    assert (package / name).is_file(), name
assert 'cgpy._cuda' not in sys.modules
for name in ('Vector', 'DenseMatrix', 'CsrMatrix', 'PreparedSolver',
             'Workspace', 'Session', 'SupportsCudaArrayInterface',
             'prepare', 'prepare_host',
             'solve', 'solve_into'):
    assert hasattr(cuda, name), name
assert 'SupportsCudaArrayInterface' in cuda.__all__
for constructor, message in (
    (cuda.PreparedSolver, 'use prepare or prepare_host'),
    (cuda.DenseMatrix, 'use DenseMatrix.from_host or DenseMatrix.from_device'),
    (cuda.CsrMatrix, 'use CsrMatrix.from_host or CsrMatrix.from_device'),
):
    assert not inspect.signature(constructor).parameters, constructor
    try:
        constructor()
    except TypeError as error:
        assert str(error) == message, (constructor, error)
    else:
        raise AssertionError(f'{constructor.__name__} accepted direct construction')
assert 'cgpy._cuda' not in sys.modules
print(f'Typed consumer package: {package}; CUDA extension remains unloaded.')
"""
        python = str(options.python.absolute())
        if options.package_root is not None:
            expected = str(consumer)
        else:
            expected = subprocess.check_output(
                [python, "-c", "import sys; print(sys.prefix)"],
                cwd=consumer,
                env=environment,
                text=True,
            ).strip()
        subprocess.run(
            [python, "-c", probe, expected],
            cwd=consumer,
            env=environment,
            check=True,
        )
        config = consumer / "mypy.ini"
        config.write_text(
            "[mypy]\n"
            "python_version = 3.12\n"
            "strict = True\n"
            "warn_unused_ignores = True\n"
            "show_error_codes = True\n"
            "follow_imports = silent\n"
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "mypy",
                "--config-file",
                str(config),
                "--python-executable",
                python,
                "--no-incremental",
                "--cache-dir",
                str(consumer / ".mypy_cache"),
                "valid.py",
                "invalid.py",
            ],
            cwd=consumer,
            env=environment,
            check=True,
        )


if __name__ == "__main__":
    main()
