# Security

What TokenCollider's local server accepts and refuses.

[← README](../README.md)

`tokencollider view` starts an HTTP server on 127.0.0.1. The viewport uses it
to talk to the Python side, and it serves the browser viewport itself.

## What it defends against

- **Web pages you visit.** A browser lets any page send requests to
  127.0.0.1. A page can also use DNS rebinding: point its own domain at
  127.0.0.1, so the browser treats the server as part of that page and lets it
  read the replies.
- **Other programs and user accounts on the machine.** They can all reach a
  port on 127.0.0.1.

Code already running as your user is out of scope. No local tool can stop it.

## What it enforces

- **127.0.0.1 only.** There's no option to listen on the network.
- **A token on every API request.** Each session makes a random token
  (`TOKENCOLLIDER_TOKEN` sets a fixed one) and refuses any request without it
  in the `X-TokenCollider-Token` header. This keeps out other programs and web
  pages. The token is never put on a command line, where other accounts
  could read it:
  - The desktop viewport gets it in its environment.
  - The browser gets it in the URL after `#`, which browsers never send to a
    server. `view` writes a redirect page only you can read (mode 0600), opens
    that file, and deletes it when the server stops.
- **The `Host` header must be `127.0.0.1:<port>` or `localhost:<port>`.** A
  rebinding page's requests carry its own domain there, so they're refused.
- **The `Origin` header,** when present, must be the server's own page.
- **No CORS headers,** so a browser can't read replies meant for another page.
- **POST bodies must be JSON** (`Content-Type: application/json`). A plain
  HTML form can't send that type to another site, so this also blocks forged
  form posts.
- **Request size and time limits.** Bodies over 1 MiB are refused, idle
  connections close after 60 seconds, and at most 32 are served at once.
- **The browser viewport's files are served by exact name** from a list made
  at startup. A request path never reaches the filesystem. These files need no
  token, since the page holds no secret, but they do need a valid `Host`.
- **Exports stay in one folder.** You choose it at launch (`--export-dir`,
  default `exports/` in TokenCollider's home). A request can only name a path
  inside it. Absolute paths, `..` and symlinks that lead out are refused.
- **Saved universe names lose any folder part,** so a request can't choose
  where they're written.
- **Universe files only point to other universe files.** A universe's header
  can name its parent universe by path. That path is followed only if the
  file there is itself a universe.

Refusals return 403 (not allowed) or 400 (bad request) straight away.

`tests/smoke_layout.py` covers these in `test_server_refuses_hostile_requests`
and the tests after it.

## Known gaps

- The cache, exports and saved universes hold every phrase in plain text, and
  vectors can be decoded back to text. The README says where they're stored;
  the viewport doesn't mention it yet.
- No rate limiting. A local program could keep the GPU busy.
