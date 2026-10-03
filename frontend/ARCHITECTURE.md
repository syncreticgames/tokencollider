# Frontend architecture

Where things live and why, so a change lands in one file instead of five.

## The shape

```
frontend/
  Main.gd            orchestration: input, scene wiring, the viewport itself
  lib/               pure logic. No scene, no node, no sidecar.
    Oklab.gd         sRGB <-> Oklab and the z-score mapping
    Matcher.gd       autocomplete ranking
    ShapeAtlas.gd    sprite textures, distance fields, density fills
    ViewState.gd     layer band, axes chart, scaling, and the view signature
    Config.gd        durable settings in user://settings.cfg
    Sidecar.gd       all HTTP and URL knowledge
  viewport/          world components. Take a Context, emit intent.
    Context.gd       the shared seam
    LandmarkFactory.gd  builds sprites, digits, landmarks
    MassView.gd      blend weights as stellar mass and tethers
    Constellations.gd   per-landmark trails across layers
    Ghosts.gd        interrogation results as in-world markers
  ui/                screens, one scene + script each
    MenuOverlay.gd   routes between the three screens; M toggles it
    MainMenu.tscn    navigation only; emits what the user asked for
    SettingsPanel.tscn  reads and writes Config, owns no state
    CreditsPanel.tscn   project credits + the engine's licence notice
  tests/             run_tests.gd plus one suite per module
```

## The three rules

**1. `lib/` is pure.** No `Node`, no scene, no network. If it needs any of
those it is not a lib module. That keeps it testable headlessly, which is how
the tests run, on a desktop and in CI alike.

**2. Components receive state and emit intent.** A `viewport/` component
takes a `Context` and reads from it; when it wants the world changed it emits
a signal and lets `Main.gd` decide. No component calls back into the scene.
That is the property that lets one be moved, replaced, or tested alone.
Components that reach into `Main` for shared state can't be separated: moving
one pulls along everything else that reaches for the same things.

**3. Shared goes on the Context; private stays private.** If two components
need a value it belongs on `Context.gd`. If one does, it belongs to that
component. `WORLD_TARGET` and `rms_spread` are there because the layout scaler
and the trails both size the world with them, and two copies of a scale
function is how two views of one universe end up different sizes.

## The menu is an overlay, not the boot scene

`MenuOverlay` mounts over the viewport on **M** rather than replacing
`Main.tscn` as the entry point. That keeps `tokencollider view` behaving exactly as it
always has and means a broken menu can never stop someone reaching the tool.
Promote it to the boot scene once it has been used in anger, not before.

While it is open the viewport ignores its own shortcuts, or `R` would refresh
and `X` would export behind an open dialog.

Settings currently exposes two of the nine keys in `Config.DEFAULTS` (port and
normalized scaling) plus a reset. The rest persist and are readable but have
no UI yet. That is scaffolding, not an oversight.

## Things that will bite you

**The view signature is load-bearing.** A cursor dropped in one view is
meaningless in another: layer bands are different subspaces, and world and
local axes are different charts of the same layer. Reprojecting across either
is silently WRONG, not approximate, and the symptom is an export from
coordinates the user never selected. Compare `ViewState.signature()` with
`==`; never translate a position between two of them.

**Never bulk find-and-replace across a `.gd` file.** A substitution like
`landmarks -> ctx.landmarks` also rewrites the string literal `"/landmarks"`
(an API endpoint) and the JSON key `"n_landmarks"`. Neither fails to compile.
Both fail at runtime, far from the edit. Substitute inside code spans only, or
do it by hand.

**Godot's `class_name` is not registered under `--script`.** Modules are
loaded with `preload` so headless tests work, and so a reader can see the
dependency at the top of the file.

## Running the tests

```
godot --headless --path frontend --script tests/run_tests.gd
```

Exit code is non-zero on failure. A suite that fails to COMPILE is reported as
a failure rather than skipped, because a silently skipped suite looks exactly
like a passing one.
