# Frontend architecture

Where the viewport's code lives, and the rules that keep it separable.

## Files

```
frontend/
  Main.gd            input, scene wiring, the viewport itself
  FlyCamera.gd       the free-fly camera
  lib/               plain logic: no scene, no nodes, no network
    Oklab.gd         sRGB <-> Oklab and the color mapping
    Matcher.gd       autocomplete ranking
    ShapeAtlas.gd    sprite textures and density fills
    ViewState.gd     layer range, axes, scaling, and the view signature
    RequestGate.gd   drops replies that arrive for an old view
    Config.gd        saved settings, in user://settings.cfg
    Sidecar.gd       every HTTP request and URL
  viewport/          things in the 3D world
    Context.gd       state the components share
    LandmarkFactory.gd  builds sprites, labels and landmarks
    MassView.gd      blend weights, drawn as size and tethers
    Constellations.gd   each landmark's trail across layers
    Ghosts.gd        query results, drawn in the world
  ui/                one scene and script per screen
    MenuOverlay.gd   switches between the screens; M opens it
    MainMenu.tscn    the menu
    SettingsPanel.tscn  reads and writes Config
    CreditsPanel.tscn   credits and Godot's license notice
  tests/             run_tests.gd and one test file per module
```

## Rules

1. **`lib/` stays plain.** No nodes, scenes or network, so it can be tested
   headless.
2. **Components read a Context and emit signals.** A `viewport/` component
   never calls into the scene. When it wants something changed, it emits a
   signal and `Main.gd` acts on it. That keeps each one movable and testable
   on its own.
3. **Shared values live on the Context.** If two components need a value, it
   goes on `Context.gd`. If one does, it stays in that component. For example,
   the layout and the trails both size the world with `WORLD_TARGET` and
   `rms_spread`, so there's one copy of each.

The menu is an overlay that M opens over the viewport. While it's open, the
viewport ignores its own keys and the camera stands still.

## Pitfalls

- **Positions belong to one view.** A position in one layer range or set of
  axes means nothing in another, and converting it gives wrong coordinates
  without any error. Compare views with `ViewState.signature()` and `==`.
- **Don't find-and-replace across a `.gd` file.** Replacing `landmarks` with
  `ctx.landmarks` would also change the endpoint `"/landmarks"` and the JSON
  key `"n_landmarks"`. Both still compile and only fail at runtime.
- **`class_name` isn't registered under `--script`.** Modules are loaded with
  `preload`, so headless tests work and each file's dependencies are listed at
  the top.

## Tests

```
godot --headless --path frontend --script tests/run_tests.gd
```

The exit code is non-zero if any check fails, a test hits a script error, or
a test file doesn't compile. Tests write their own settings file, so they
never touch your saved settings.
