"""Per-model knowledge: how a checkpoint wants its prompts wrapped, which
hidden-state layers its sampler actually reads, and where its files live.

None of it is code. Two files, split along what belongs in version control:

- `tokencollider/builtin_profiles.yaml` carries what is verified about each known
  model: the prompt scaffold, the sampler layers, whether exports drop the
  template prefix, which band is worth charting. Findings about a model,
  the same for everyone.
- `profiles.yaml` in TokenCollider's home: beside the repo in a checkout, the user
  data folder when installed (see `paths.py`; gitignored; see `profiles.example.yaml`)
  supplies where the files are on one machine, and may define new profiles,
  from scratch or by extending another:

    profiles:
      krea2-finetune:
        extends: krea2
        model: /models/qwen3vl-4b-finetune

A finding that turns out wrong is an edit to a data file, not to how the
tool works.

Two settings that used to be one. `layer` is the chart the viewport
navigates in, a single layer or a band. `sampler_layers` is what an export has
to carry. They coincide for a model that reads one layer, and not for one
that reads several.
"""

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import paths


@dataclass(frozen=True)
class ModelProfile:
    name: str
    # Context the phrase is embedded inside. Pooling covers only the phrase's
    # own token span, so the scaffold conditions attention without
    # contributing tokens.
    template: str
    # Layers a conditioning must carry. None means "whatever `layer` resolves
    # to", the single-layer case.
    sampler_layers: tuple[int, ...] | None = None
    # The layer or band the viewport charts (--layer, TOKENCOLLIDER_<PROFILE>_LAYER).
    layer: str = "last"
    # Viewport slider bounds: the band worth scrubbing. Layers outside are not
    # hidden data, just hidden controls.
    layer_min: int | None = None
    layer_max: int | None = None
    # Where this model's files are. Machine facts, so they stay out of the
    # repo and come from profiles.yaml (or the scoped environment variable).
    model_path: str | None = None
    config_dir: str | None = None
    # Prefix for this profile's environment variables, e.g. TOKENCOLLIDER_KREA2_MODEL,
    # TOKENCOLLIDER_KREA2_CONFIG_DIR, TOKENCOLLIDER_KREA2_LAYER. Every setting is read under this
    # prefix and nowhere else.
    env_prefix: str = "CX"
    # Where this profile came from, for the banner: "built-in" or a file path.
    origin: str = "built-in"
    # Whether the sampler is handed the prompt tokens onward, with the
    # template scaffold that precedes them dropped. ComfyUI's Krea 2 encoder
    # does this (it finds the <|im_start|> opening the user turn, adds 3, and
    # slices there); its Z-Image encoder does not. A per-model convention, so
    # it lives here rather than being applied everywhere.
    trim_template_prefix: bool = False
    # Fallback source for config + tokenizer, resolved against the local HF
    # cache only. A config_dir overrides.
    config_repo: str | None = None
    # ai-toolkit's `arch` for this model, which is also the
    # text_embedding_space_version its cache filenames are hashed with.
    # None means the bridge has no contract for this model.
    trainer_arch: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


BUILTIN_FILE = Path(__file__).resolve().parent / "builtin_profiles.yaml"

# profiles.yaml lives in TokenCollider's home (beside the repo in a checkout,
# the user data folder when installed; see paths.py). $TOKENCOLLIDER_PROFILES
# points somewhere else.
PROFILES_FILE = paths.home() / "profiles.yaml"

# Fields a profiles.yaml entry may set. "model" maps to model_path because
# that is what it reads like in a config file.
YAML_FIELDS = {
    "model": "model_path",
    "config_dir": "config_dir",
    "config_repo": "config_repo",
    "layer": "layer",
    "layer_min": "layer_min",
    "layer_max": "layer_max",
    "template": "template",
    "sampler_layers": "sampler_layers",
    "trim_template_prefix": "trim_template_prefix",
    "env_prefix": "env_prefix",
    "trainer_arch": "trainer_arch",
    "notes": "notes",
}


def profiles_path() -> Path:
    return Path(os.environ.get("TOKENCOLLIDER_PROFILES") or PROFILES_FILE)


def _apply(base: ModelProfile, name: str, entry: dict, origin: str) -> ModelProfile:
    unknown = sorted(set(entry) - set(YAML_FIELDS) - {"extends"})
    if unknown:
        raise SystemExit(
            f"[tokencollider] profile {name!r} in {origin} sets unknown field(s) "
            f"{', '.join(unknown)}. Known: {', '.join(sorted(YAML_FIELDS))}"
        )
    changes = {YAML_FIELDS[k]: v for k, v in entry.items() if k in YAML_FIELDS}
    if "sampler_layers" in changes:
        changes["sampler_layers"] = parse_sampler_layers(changes["sampler_layers"])
    if "notes" in changes:
        changes["notes"] = tuple(changes["notes"] or ())
    return replace(base, name=name, origin=origin, **changes)


def load(path: Path | None = None) -> dict[str, ModelProfile]:
    """The built-in profiles, overlaid and extended by profiles.yaml if
    present. `default:` is honoured from the user's file only."""
    registry: dict[str, ModelProfile] = {}
    _overlay(registry, BUILTIN_FILE, "built-in")
    path = Path(path) if path is not None else profiles_path()
    if path.exists():
        default = _overlay(registry, path, str(path))
        if default is not None and default not in registry:
            raise SystemExit(f"[tokencollider] {path}: default profile {default!r} is not defined")
        if default:
            registry["__default__"] = registry[default]
    return registry


def _overlay(registry: dict, path: Path, origin: str):
    """Apply one profiles file to the registry. Returns its `default:`."""
    import yaml

    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise SystemExit(f"[tokencollider] cannot parse {path}: {e}")
    if not isinstance(doc, dict):
        raise SystemExit(f"[tokencollider] {path} must be a mapping with a 'profiles' key")
    entries = doc.get("profiles") or {}
    if not isinstance(entries, dict):
        raise SystemExit(f"[tokencollider] {path}: 'profiles' must be a "
                         "mapping of name -> settings")

    pending = dict(entries)
    # Resolve `extends` by repeated passes, so order in the file does not
    # matter. A cycle stops making progress and is reported rather than hung on.
    while pending:
        progressed = False
        for name in list(pending):
            entry = pending[name] or {}
            if not isinstance(entry, dict):
                raise SystemExit(f"[tokencollider] {path}: profile {name!r} must be a mapping")
            parent_name = entry.get("extends", name if name in registry else None)
            if parent_name is None:
                # A profile from scratch: everything else has a default, the
                # template does not.
                if "template" not in entry:
                    raise SystemExit(
                        f"[tokencollider] {path}: profile {name!r} is new, so it needs a "
                        "'template' or 'extends' naming a base"
                        + (f" ({', '.join(sorted(registry))})" if registry else ""))
                base = ModelProfile(name=name, template=entry["template"],
                                    env_prefix=f"TOKENCOLLIDER_{_slug(name)}")
            else:
                if parent_name in pending and parent_name != name:
                    continue  # base itself is still being resolved
                if parent_name not in registry:
                    raise SystemExit(
                        f"[tokencollider] {path}: profile {name!r} extends unknown profile "
                        f"{parent_name!r}"
                    )
                base = registry[parent_name]
                if parent_name != name:
                    base = replace(base, env_prefix=f"TOKENCOLLIDER_{_slug(name)}")
            registry[name] = _apply(base, name, entry, origin)
            del pending[name]
            progressed = True
        if not progressed:
            raise SystemExit(
                f"[tokencollider] {path}: circular 'extends' among "
                f"{', '.join(sorted(pending))}"
            )
    return doc.get("default")


def builtins() -> dict[str, ModelProfile]:
    """The built-in profiles alone, without the user's overlay: the
    reference conventions a contract (such as a trainer cache) is checked
    against."""
    registry: dict[str, ModelProfile] = {}
    _overlay(registry, BUILTIN_FILE, "built-in")
    return registry


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).upper()


def get(name: str | None = None, path: Path | None = None) -> ModelProfile:
    """Look up a profile by name. Falls back to $TOKENCOLLIDER_PROFILE, then the
    profiles.yaml `default`. There is no built-in default model."""
    registry = load(path)
    if name is None:
        name = os.environ.get("TOKENCOLLIDER_PROFILE")
    if name is None:
        fallback = registry.get("__default__")
        if fallback is None:
            known = ", ".join(sorted(k for k in registry if not k.startswith("__")))
            raise SystemExit(
                "[tokencollider] no model profile chosen: --profile, $TOKENCOLLIDER_PROFILE, or "
                f"'default:' in {profiles_path()}. Known: {known}")
        name = fallback.name
    try:
        return registry[name]
    except KeyError:
        known = ", ".join(sorted(k for k in registry if not k.startswith("__")))
        raise SystemExit(f"[tokencollider] unknown model profile {name!r}. Known: {known}")


def resolve(profile: ModelProfile, name: str, flag=None, default=None):
    """A setting's value and where it came from.

    Order: the explicit flag, then this profile's own variable
    (TOKENCOLLIDER_KREA2_MODEL), then the profile's default. There is deliberately no
    shared TOKENCOLLIDER_MODEL / TOKENCOLLIDER_LAYER / TOKENCOLLIDER_TEMPLATE tier.

    A global cannot say which model it was set for, and the two encoders this
    tool runs have identical text-tower shapes (36 layers at 2560). The wrong
    weights under the right config, or a leftover template from the other
    model, load without one missing key and are wrong in exactly the ways no
    error catches. Scoping every variable to its profile removes the class of
    mistake rather than warning about it.
    """
    if flag is not None:
        return flag, "flag"
    var = f"{profile.env_prefix}_{name}"
    value = os.environ.get(var)
    if value not in (None, ""):
        return value, var
    return default, f"profile:{profile.name}"


def resolve_int(profile: ModelProfile, name: str, default=None):
    value, source = resolve(profile, name, None, default)
    if value in (None, ""):
        return None, source
    return int(value), source


def require(profile: ModelProfile, name: str, value, what: str):
    """Fail loudly rather than guess. A profile with no path for something it
    needs is a configuration the user has not finished, not a default to
    invent."""
    if value in (None, ""):
        raise SystemExit(
            f"[tokencollider] the {profile.name!r} profile has no {what}. Add a "
            f"'{name.lower()}:' line for it in {profiles_path()}, or set "
            f"{profile.env_prefix}_{name}."
        )
    return value


def parse_sampler_layers(spec):
    """'2,5,8' -> (2, 5, 8). None or '' -> None (use the resolved layer)."""
    if spec is None or isinstance(spec, tuple):
        return spec
    if isinstance(spec, (list, set)):
        return tuple(sorted(int(v) for v in spec))
    spec = str(spec).strip()
    if not spec:
        return None
    return tuple(sorted(int(part) for part in spec.split(",") if part.strip()))
