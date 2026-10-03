"""Where TokenCollider keeps its own files: the embedding cache, profiles.yaml,
exports, and saved universes.

A source checkout keeps them beside the repo, as it always has. An installed
package keeps them in the user's data folder, never inside the Python
environment, since the cache and exports carry every embedded phrase in
plaintext. TOKENCOLLIDER_HOME overrides both."""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def is_checkout() -> bool:
    """True when running from a source tree (including an editable install),
    false from a wheel, whose parent directory is site-packages."""
    return (REPO / "pyproject.toml").is_file() and (REPO / "tokencollider").is_dir()


def user_data_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        return (Path(base) if base else Path.home() / "AppData" / "Local") / "TokenCollider"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "TokenCollider"
    # XDG says a relative XDG_DATA_HOME is invalid and must be ignored.
    base = os.environ.get("XDG_DATA_HOME")
    base = Path(base) if base and Path(base).is_absolute() else Path.home() / ".local" / "share"
    return base / "tokencollider"


def home() -> Path:
    env = os.environ.get("TOKENCOLLIDER_HOME")
    if env:
        return Path(env).expanduser()
    return REPO if is_checkout() else user_data_dir()
