extends RefCounted
## Shared viewport state, handed to components instead of reached for.
##
## The reason the old Main.gd could not be split: constellations, ghosts and
## the mass view all needed the same handful of things (the landmark nodes,
## the world scale, a way to call the sidecar, a way to report), and each one
## reached into Main for them. Any attempt to move one file pulled the rest
## with it.
##
## This is the seam. Components take a Context and use only what it exposes,
## so a component can be constructed in a test, reused in a second viewport,
## or moved to another file without touching anything else. The rule for
## adding to this: if two components need it, it belongs here; if one does, it
## belongs to that component.

## The farthest landmark lands about this many world units out. Two components
## size themselves against it, so it lives here rather than being copied.
const WORLD_TARGET := 20.0

## The landmark nodes by phrase. Components read; the field owns writes.
var landmarks := {}

## Layout units to world units. THE one copy: the layout placer, the cursor,
## the ghosts and the trails all read it, and the trails write it. A second
## copy in Main.gd is what made trails render at a scale nothing else used.
var world_scale := 1.0

## Depth-stable scale from the trajectory extent, 0 when no trails are drawn.
## Constellations computes it; the layout placer reads it in absolute mode so
## spheres land exactly on their trail lines. Shared, so it lives here.
var locked_scale := 0.0

## What the user is looking at (lib/ViewState.gd). Components compare
## signatures to decide whether cached positions are still meaningful.
var view: RefCounted = null

## `await ctx.api.call(path, body)` -> parsed response or null. Injected so a
## component never owns transport and a test never needs a live sidecar.
var api: Callable = func(_path: String, _body = null): return null

## One line of human-readable feedback. A Callable, not a Label, so the same
## component works in a viewport, a settings preview, or a headless test.
var report: Callable = func(_what: String) -> void: pass


func signature() -> String:
	return view.signature() if view != null else ""


static func rms_spread(entries: Array) -> float:
	## Root-mean-square spatial radius over entries carrying 6D "coords".
	##
	## Static and here rather than on either caller: the layout scaler and the
	## trails both size the world with it, and two copies of a scale function
	## is how two views of the same universe end up different sizes.
	if entries.is_empty():
		return 0.0
	var sq := 0.0
	for e in entries:
		var c: Array = e["coords"]
		sq += c[0] * c[0] + c[1] * c[1] + c[2] * c[2]
	return sqrt(sq / entries.size())
