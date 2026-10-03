extends RefCounted
## sRGB <-> Oklab, and the z-score mapping the sidecar colours points with.
##
## Oklab is Björn Ottosson's colour space, and the matrices here are his, from
## https://bottosson.github.io/posts/oklab/ (the same reference oklab.py cites).
##
## Mirrors `tokencollider/oklab.py`, which must produce the same numbers or a picked
## colour and the point it selects drift apart, invisibly, until an export
## lands somewhere the user did not aim.
##
## The two are NOT plain inverses, deliberately. `hex_to_zscores` is the
## inverse of `zscores_to_hex` only WITHIN GAMUT: the corners of the z-cube
## fall outside sRGB and clip on the way out. And Python returns raw z on the
## way back while this clamps to the displayable range, because a colour the
## sidecar could never paint should still select the nearest point it can.
## `tests/test_oklab.gd` checks agreement against a fixture the Python
## generates, which is the only comparison that tests one contract.
##
## Pure and static on purpose: no scene, no state, no node. That is what makes
## it testable headlessly and safe to call from anywhere.
##
## Loaded with `preload`, not `class_name`. A global class name is not
## registered when Godot runs with `--script`, so a headless test that relied
## on it would fail for a reason that has nothing to do with the code. An
## explicit preload also states the dependency where a reader can see it.

const L_CENTER := 0.72
const L_SPAN := 0.10
const AB_SPAN := 0.12
const Z_CLAMP := 2.0


static func srgb_to_linear(ch: float) -> float:
	return ch / 12.92 if ch <= 0.04045 else pow((ch + 0.055) / 1.055, 2.4)


static func srgb_to_oklab(c: Color) -> Vector3:
	var r := srgb_to_linear(c.r)
	var g := srgb_to_linear(c.g)
	var b := srgb_to_linear(c.b)
	var l := 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
	var m := 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
	var s := 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
	var l_ := pow(l, 1.0 / 3.0)
	var m_ := pow(m, 1.0 / 3.0)
	var s_ := pow(s, 1.0 / 3.0)
	return Vector3(
		0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
		1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
		0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_)


static func color_to_zscores(c: Color) -> Vector3:
	## Clamped to the same +/-Z_CLAMP the sidecar renders with, so the picked
	## colour and the selected point can never disagree.
	var lab := srgb_to_oklab(c)
	return Vector3(
		clampf((lab.x - L_CENTER) / L_SPAN * Z_CLAMP, -Z_CLAMP, Z_CLAMP),
		clampf(lab.y / AB_SPAN * Z_CLAMP, -Z_CLAMP, Z_CLAMP),
		clampf(lab.z / AB_SPAN * Z_CLAMP, -Z_CLAMP, Z_CLAMP))
