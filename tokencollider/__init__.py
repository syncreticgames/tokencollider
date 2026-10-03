"""TokenCollider: navigate a text encoder's embedding space by landmarks and universes."""

from importlib.metadata import PackageNotFoundError, version

# One source for the version: pyproject.toml, read back from the installed
# package's metadata (a `uv sync` checkout has it too), so the two can't drift.
try:
    __version__ = version("tokencollider")
except PackageNotFoundError:  # running from source that was never installed
    __version__ = "0+unknown"
