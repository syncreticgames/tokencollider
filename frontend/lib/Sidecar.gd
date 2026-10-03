extends Node
## The sidecar HTTP client. Everything that knows a URL lives here.
##
## It reports failures by signal and owns no UI. So a request can be tested
## without a scene, and each caller decides what a failure looks like: a main
## menu can show a connection dot and the viewport a status line, from the
## same client.

signal request_failed(what: String)   ## human-readable, safe to show a user
signal unreachable                    ## the sidecar is not answering at all

const TOKEN_HEADER := "X-TokenCollider-Token"

var host := "127.0.0.1"
var port := 8765
## The per-session token the sidecar requires on every API request (see
## docs/security.md). `tokencollider view` hands it over out of band.
var token := ""


func base_url() -> String:
	return "http://%s:%d" % [host, port]


func auth_headers() -> PackedStringArray:
	return PackedStringArray(["%s: %s" % [TOKEN_HEADER, token]])


func apply_launch(env: Dictionary, page_url: String = "") -> void:
	## Where `tokencollider view` says the sidecar is, overriding saved
	## settings: that reflects reality, the settings are what the user last
	## chose. In a browser the page came FROM the sidecar, so its own URL is
	## the answer, with the token in the fragment. On desktop, the env.
	if page_url != "":
		var rest := page_url.split("://", true, 1)[1]
		var authority := rest.split("/", true, 1)[0]
		host = authority.split(":")[0]
		var p := authority.split(":")
		port = p[1].to_int() if p.size() > 1 else 80
		var hash_at := page_url.find("#")
		if hash_at >= 0:
			for pair in page_url.substr(hash_at + 1).split("&"):
				if pair.begins_with("token="):
					token = pair.substr(6).uri_decode()
		return
	if str(env.get("TOKENCOLLIDER_PORT", "")).is_valid_int():
		port = str(env["TOKENCOLLIDER_PORT"]).to_int()
	token = str(env.get("TOKENCOLLIDER_TOKEN", token))


## How long a request may take before it counts as lost. Generous on
## purpose: the first visit to a layer embeds every landmark there, and an
## export cooks through the encoder (a full sweep, at every depth). The point
## is that a hung sidecar no longer leaves the viewport waiting forever.
const TIMEOUT_DEFAULT := 600.0
const TIMEOUT_EXPORT := 3600.0


static func timeout_for(path: String) -> float:
	return TIMEOUT_EXPORT if path.begins_with("/export") else TIMEOUT_DEFAULT


func request_bytes(path: String, timeout := 30.0) -> Variant:
	## GET raw bytes (an image thumbnail). Returns them, or null after
	## emitting the reason, the same way request() reports.
	var http := HTTPRequest.new()
	http.timeout = timeout
	add_child(http)
	if http.request(base_url() + path, auth_headers()) != OK:
		http.queue_free()
		request_failed.emit("couldn't start loading %s" % path)
		return null
	var res: Array = await http.request_completed
	http.queue_free()
	if res[0] != HTTPRequest.RESULT_SUCCESS or res[1] != 200:
		request_failed.emit("couldn't load %s (result %d, status %d)" % [path, res[0], res[1]])
		return null
	return res[3]


func request(path: String, body = null) -> Variant:
	## GET when body is null, POST as JSON otherwise. Returns the parsed
	## response, or null after emitting the reason.
	##
	## The JSON content type is not decoration: the sidecar refuses any POST
	## without it, which is what stops a web page from driving this API
	## (see docs/security.md). Do not "simplify" it away.
	var http := HTTPRequest.new()
	http.timeout = timeout_for(path)
	add_child(http)
	var err: int
	if body == null:
		err = http.request(base_url() + path, auth_headers())
	else:
		var headers := auth_headers()
		headers.append("Content-Type: application/json")
		err = http.request(base_url() + path, headers,
			HTTPClient.METHOD_POST, JSON.stringify(body))
	if err != OK:
		http.queue_free()
		request_failed.emit("request failed to start (%d)" % err)
		return null
	var res: Array = await http.request_completed
	http.queue_free()
	if res[0] == HTTPRequest.RESULT_TIMEOUT:
		request_failed.emit("no answer from the sidecar in %d s; it may be busy or stuck (see its terminal)" % int(timeout_for(path)))
		return null
	if res[0] != HTTPRequest.RESULT_SUCCESS:
		unreachable.emit()
		request_failed.emit("sidecar unreachable")
		return null
	var parsed = JSON.parse_string(res[3].get_string_from_utf8())
	if res[1] != 200:
		# The sidecar answers 403 for a refused request and 400 for a bad one;
		# both carry an {"error": ...} body worth showing verbatim.
		var detail = parsed.get("error", parsed) if parsed is Dictionary else parsed
		request_failed.emit("sidecar error %d: %s" % [res[1], str(detail)])
		return null
	return parsed
