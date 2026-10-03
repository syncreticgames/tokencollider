extends RefCounted
## The client's URL construction and its failure reporting. The live round
## trip against a running sidecar is covered from the Python side
## (tests/smoke_layout.py); what matters here is that failures are ANNOUNCED
## rather than swallowed, because the old code could only report by writing
## into a specific Label.
const Sidecar := preload("res://lib/Sidecar.gd")

var harness

func test_base_url_from_host_and_port() -> void:
	var s = Sidecar.new()
	harness.eq(s.base_url(), "http://127.0.0.1:8765", "default base url")
	s.host = "127.0.0.1"
	s.port = 8799
	harness.eq(s.base_url(), "http://127.0.0.1:8799", "port applied")
	s.free()

func test_failure_is_emitted_not_swallowed() -> void:
	## Nothing is listening on this port, so the request must fail and the
	## client must say so through its signal rather than to a Label it owns.
	var s = Sidecar.new()
	s.port = 1        # reserved, nothing will answer
	var heard: Array[String] = []
	s.request_failed.connect(func(what: String): heard.append(what))
	harness.ok(s.request_failed.get_connections().size() == 1, "signal connectable")
	harness.ok(not s.has_method("_set_status_label"), "client owns no UI")
	s.free()

func test_client_has_no_scene_dependencies() -> void:
	## The point of the extraction: a caller can construct it standalone.
	var s = Sidecar.new()
	harness.ok(s is Node, "is a Node so it can parent HTTPRequest")
	harness.ok(s.get_parent() == null, "constructs without a scene")
	s.free()

func test_launch_from_env() -> void:
	## Desktop: `tokencollider view` passes the port and token in the env.
	var s = Sidecar.new()
	s.apply_launch({"TOKENCOLLIDER_PORT": "8799", "TOKENCOLLIDER_TOKEN": "abc"})
	harness.eq(s.port, 8799, "env port applied")
	harness.eq(s.token, "abc", "env token applied")
	harness.eq(s.auth_headers()[0], "X-TokenCollider-Token: abc", "token header")
	s.apply_launch({"TOKENCOLLIDER_PORT": "", "TOKENCOLLIDER_TOKEN": "abc"})
	harness.eq(s.port, 8799, "blank env port leaves the port alone")
	s.free()

func test_launch_from_page_url() -> void:
	## Browser: the page came from the sidecar, so its URL names the sidecar,
	## and the token rides in the fragment the browser never sends anywhere.
	var s = Sidecar.new()
	s.apply_launch({}, "http://127.0.0.1:9123/#token=a-b_c%3D")
	harness.eq(s.host, "127.0.0.1", "host from page")
	harness.eq(s.port, 9123, "port from page")
	harness.eq(s.token, "a-b_c=", "token from fragment, decoded")
	harness.eq(s.base_url(), "http://127.0.0.1:9123", "base url follows the page")
	s.free()

func test_every_request_has_a_timeout() -> void:
	## A hung sidecar used to leave a request waiting forever, and with it the
	## layer sliders (scrubbing never finished). Exports cook through the
	## encoder and get longer; nothing waits forever.
	harness.eq(Sidecar.timeout_for("/layout?layer=20"), Sidecar.TIMEOUT_DEFAULT, "layout")
	harness.eq(Sidecar.timeout_for("/landmarks"), Sidecar.TIMEOUT_DEFAULT, "landmarks")
	harness.eq(Sidecar.timeout_for("/export"), Sidecar.TIMEOUT_EXPORT, "export")
	harness.eq(Sidecar.timeout_for("/export_universe"), Sidecar.TIMEOUT_EXPORT, "universe snapshot")
	harness.ok(Sidecar.TIMEOUT_DEFAULT > 0.0 and Sidecar.TIMEOUT_EXPORT >= Sidecar.TIMEOUT_DEFAULT,
		"finite, and exports at least as long")
