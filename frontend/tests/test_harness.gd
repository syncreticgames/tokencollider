extends RefCounted
## Proves the runner itself reports passes and failures, so a green run
## means the tests actually executed rather than the loader silently
## skipping them.
var harness

func test_harness_runs_and_counts() -> void:
	harness.ok(true, "true is ok")
	harness.eq(2 + 2, 4, "arithmetic")
	harness.close(0.1 + 0.2, 0.3, 1e-9, "float compare")
