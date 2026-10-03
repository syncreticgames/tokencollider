# Security

What the sidecar exposes, and what it refuses.

[← TokenCollider README](../README.md)

TokenCollider runs a local HTTP sidecar so the Godot viewport can talk to the
Python side. A process that binds a port is the part of this project that can
hurt someone who installs it, so the rules are narrow and enforced in code
rather than by convention.

## Threat model

Two attackers worth designing against on a desktop:

- **Any page the user visits.** A browser can POST to `127.0.0.1` from any
  origin. It cannot read the response without CORS, but a write-only request
  is enough to do damage.
- **Any process on the box.** Loopback is not an authorisation boundary. Every
  local process and user account can reach the port.

Not in scope: an attacker who already runs code as the user. Nothing a local
tool does survives that.

## What is enforced

- **Loopback only.** The listener is hardcoded to `127.0.0.1`. There is no
  flag to change it, because there is no version of this tool that should be
  reachable from the network.
- **No CORS headers, ever.** The server answers no preflight, so a browser
  cannot read anything it gets back.
- **POST requires `Content-Type: application/json`.** A cross-origin form POST
  can only send `text/plain`, `application/x-www-form-urlencoded`, or
  `multipart/form-data`. Requiring JSON forces a preflight, which fails. This
  is what blocks CSRF, and the Godot client already sends the header.
- **Any `Origin` header is refused.** Nothing in a browser context has
  business here. Belt and braces behind the previous rule.
- **Exports are confined to one directory.** `/export` takes a filename from
  the request body, which unchecked is an arbitrary file write. The OPERATOR
  picks the root when launching (`--export-dir`, default `exports/`);
  a REQUEST only names a path relative to it. Absolute paths, `..`, and
  symlinks that escape are refused with a 400. That split is the whole
  boundary: whoever started the process is trusted, whatever opened a socket
  is not.
- **Universe snapshots keep only the stem.** `export_universe` runs
  `Path(name).stem`, so a directory component in a requested name is dropped.

Refusals answer 403 (declined on principle) or 400 (path escapes the root),
never a hang: a dead handler thread would freeze the viewport.

`tests/smoke_layout.py::test_server_refuses_hostile_requests` exercises the
form-POST, cross-origin, absolute-path and traversal cases and asserts no
stray file appears.

## Known gaps

- The cache and exports carry every embedded phrase in plaintext, and vectors
  decode back to text. `.gitignore` covers `embeddings.db` and `exports/`;
  a packaged build should say so in the UI, not only in a dotfile.
- No authentication on the port. Adding a per-session token in a header would
  close the local-process case; loopback plus the JSON requirement does not.
- No rate limiting. A local denial of service is possible and low value to an
  attacker, but a runaway client can wedge the GPU.
