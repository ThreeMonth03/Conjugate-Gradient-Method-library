"""Write JSON reports atomically without imposing an artifact schema."""

import json
import os
import pathlib
import tempfile


def write_json(data, output):
    """Replace a report only after its complete UTF-8 JSON reaches the file."""
    path = pathlib.Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf8", dir=path.parent, delete=False
        ) as stream:
            temporary = pathlib.Path(stream.name)
            json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
