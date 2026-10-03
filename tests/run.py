"""One entry point: run every test file, then measure and compare metrics.
`--update` accepts the current numbers as the baseline.

Why metrics and not just pass/fail. An autonomous loop needs a target, and a
green suite gives it nothing to steer by: a change that quietly makes every
blend worse still passes. The probes below are deterministic, model-free, and
chosen to move when something real changes, so a loop can compare against
`tests/baseline.json` and see drift the assertions were never going to catch.

Nothing here needs a GPU. Questions that do need one (which layers actually
peak per modality on Qwen3-VL-4B, whether the inserter sweep transfers) live
in tests/real_model.py and are deliberately out of the loop.
"""

import argparse
import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
BASELINE = Path(__file__).resolve().parent / "baseline.json"
SUITE = ["smoke.py", "smoke_layout.py", "smoke_conditioning.py",
         "smoke_fp8.py", "synthetic.py"]
# The Godot side has its own headless runner. Included here so one command
# gates both halves; skipped with a note if godot is not installed, because a
# contributor working on the Python side should not be blocked by it.
GODOT_SUITE = ["--headless", "--path", "frontend", "--script", "tests/run_tests.gd"]
# Fractional drift a metric may move before it is called a regression.
TOLERANCE = 0.02


def orphan_tests() -> list[str]:
    """Test functions that are defined and never called.

    Most suite files dispatch from an explicit list in `__main__`, so adding a
    `test_*` and forgetting the call leaves it silently dead: the file still
    passes, and the thing it was written to catch goes uncaught. A loop that
    trusts a green suite has to know the suite actually ran. Files that
    dispatch by scanning globals() are exempt, since they cannot orphan.
    """
    orphans = []
    for name in SUITE:
        path = Path(__file__).parent / name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        defined = {n.name for n in tree.body
                   if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")}
        src = path.read_text(encoding="utf-8")
        if "globals().items()" in src:      # auto-dispatch, cannot orphan
            continue
        called = {n.func.id for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        for missing in sorted(defined - called):
            orphans.append(f"{name}: {missing} is defined but never called")
    return orphans


SOURCE_DIRS = ["tokencollider", "comfyui_node", "trainers", "tools", "tests"]


def unencoded_text_io() -> list[str]:
    """Text read or written without `encoding=`.

    Python on Windows defaults to the locale's code page (cp1252), so a
    universe with a CJK phrase or an emoji fails to save there, and a UTF-8
    file made on Linux reads back as different phrases. Every
    `read_text`/`write_text`, and every `open` not in binary mode, names
    utf-8 explicitly.
    """
    out = []
    for d in SOURCE_DIRS:
        for path in sorted((ROOT / d).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if any(k.arg == "encoding" for k in node.keywords):
                    continue
                f = node.func
                if isinstance(f, ast.Attribute) and f.attr in ("read_text", "write_text"):
                    what = f.attr
                elif isinstance(f, ast.Name) and f.id == "open":
                    mode = node.args[1] if len(node.args) > 1 else next(
                        (k.value for k in node.keywords if k.arg == "mode"), None)
                    if isinstance(mode, ast.Constant) and "b" in str(mode.value):
                        continue
                    what = "open"
                else:
                    continue
                out.append(f"{path.relative_to(ROOT)}:{node.lineno}: {what} without encoding=")
    return out


DOC_DIR = ROOT / "docs"


def duplicate_headings() -> list[str]:
    """Repeated headings in the docs.

    An edit that inserts a section by str.replace on a heading anchor will
    duplicate the whole block if that anchor already appears twice, and the
    result reads fine locally while the file quietly doubles in size. That
    happened to a doc once and nothing caught it, so now something does.
    """
    out = []
    for md in sorted(DOC_DIR.glob("*.md")):
        seen = {}
        for line in md.read_text(encoding="utf-8").splitlines():
            if line.startswith("## ") or line.startswith("### "):
                seen[line] = seen.get(line, 0) + 1
        for head, n in seen.items():
            if n > 1:
                out.append(f"{md.name}: {n}x {head!r}")
    return out


def run_godot() -> tuple[bool, str]:
    godot = shutil.which("godot") or shutil.which("godot4")
    if godot is None:
        return True, "skip  frontend (godot not installed)"
    p = subprocess.run([godot] + GODOT_SUITE, capture_output=True, text=True,
                       cwd=ROOT, timeout=900)
    tail = [l for l in p.stdout.splitlines() if l.startswith(("ok:", "FAIL"))]
    line = tail[-1] if tail else "no result"
    passed = p.returncode == 0 and line.startswith("ok:")
    return passed, ("ok    frontend: " if passed else "FAIL  frontend: ") + line


def run_suite() -> tuple[bool, dict]:
    results = {}
    for name in SUITE:
        p = subprocess.run([sys.executable, str(Path(__file__).parent / name)],
                           capture_output=True, text=True, cwd=ROOT)
        results[name] = p.returncode == 0
        if p.returncode != 0:
            print(f"FAIL {name}\n{p.stdout[-2000:]}{p.stderr[-2000:]}")
    return all(results.values()), results


def metrics() -> dict:
    """Deterministic probes over planted geometry. Every number here has a
    reason to move only when behavior changes."""
    import numpy as np

    from tokencollider.layout import LayoutSession
    from tokencollider.synthetic import SyntheticEmbedder

    HOT = ["fire", "flame", "ember", "blaze"]
    COLD = ["ice", "frost", "snow", "chill"]
    TOOL = ["hammer", "wrench", "drill", "chisel"]

    e = SyntheticEmbedder({"hot": HOT, "cold": COLD, "tool": TOOL},
                          opposites=[("hot", "cold")])
    s = LayoutSession(e, layer=e.peak_layer())
    for p in e.phrases():
        s.add_landmark(p)
    peak = e.peak_layer()
    hot = np.mean([e.embed_layer(p, peak) for p in HOT], axis=0)
    tool = np.mean([e.embed_layer(p, peak) for p in TOOL], axis=0)
    far = s.mean + 2.0 * (hot - s.mean)
    both = s.mean + (hot - s.mean) + (tool - s.mean)

    def solve(target, **kw):
        r = s.blend_weights(s.coords_of(target), None, target, **kw)
        w = r["weights"]
        return r, {g: sum(w[p] for p in ps)
                   for g, ps in (("hot", HOT), ("cold", COLD), ("tool", TOOL))}

    free, m_free = solve(far)
    strict, m_strict = solve(far, positive=HOT, negative=COLD, strict=True)
    _, m_bent = solve(both, bends=[(TOOL, 1.0)])
    _, m_plain = solve(both)

    drift = {}
    for hi in (8, 26):
        cond, blend = s.export_conditioning(s.coords_of(hot), band=(hi, hi))
        raw = s._blend_at([hi], list(solve(hot)[0]["weights"].values()))[hi]
        top = cond[max(e.sampler_layers())]
        n = min(raw.shape[0], top.shape[0])
        drift[f"inserter_{hi}_drift"] = float(
            np.linalg.norm(top[:n] - raw[:n]) / (np.linalg.norm(raw[:n]) or 1.0))

    def tight(members, layer):
        vs = [e.embed_layer(m, layer) for m in members]
        c = np.mean(vs, axis=0)
        return float(np.mean([v @ c / (np.linalg.norm(v) * np.linalg.norm(c))
                              for v in vs]))

    pics = ["image:harbour.png", "image:kestrel.png", "image:violin.png"]
    e2 = SyntheticEmbedder({"hot": HOT, "shot": pics},
                           peaks={"text": 18, "image": 28})
    return {
        "free_error": round(free["relative_error"], 6),
        "strict_error": round(strict["relative_error"], 6),
        "free_cold_mass": round(m_free["cold"], 6),
        "free_tool_mass": round(m_free["tool"], 6),
        "strict_cold_mass": round(m_strict["cold"], 6),
        "strict_tool_mass": round(m_strict["tool"], 6),
        "bent_tool_mass": round(m_bent["tool"], 6),
        "unbent_tool_mass": round(m_plain["tool"], 6),
        "negativity_alignment": round(free["negativity"]["centroid_alignment"], 6),
        "text_peak": max(range(1, 36), key=lambda l: tight(HOT, l)),
        "image_peak": max(range(1, 36), key=lambda l: tight(pics, l)),
        **{k: round(v, 6) for k, v in drift.items()},
    }


def compare(now: dict, was: dict) -> list[str]:
    out = []
    for k, v in sorted(now.items()):
        if k not in was:
            out.append(f"NEW    {k} = {v}")
            continue
        old = was[k]
        if isinstance(v, (int, float)) and isinstance(old, (int, float)):
            span = max(abs(old), 1e-6)
            if abs(v - old) / span > TOLERANCE:
                out.append(f"DRIFT  {k}: {old} -> {v}")
        elif v != old:
            out.append(f"DRIFT  {k}: {old} -> {v}")
    for k in sorted(set(was) - set(now)):
        out.append(f"GONE   {k} (was {was[k]})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true",
                    help="write current metrics as the new baseline")
    ap.add_argument("--metrics-only", action="store_true")
    args = ap.parse_args()

    ok = True
    if not args.metrics_only:
        orphans = orphan_tests() + duplicate_headings() + unencoded_text_io()
        for line in orphans:
            print(f"ORPHAN {line}")
        ok, results = run_suite()
        for name, passed in results.items():
            print(f"{'ok  ' if passed else 'FAIL'}  {name}")
        gd_ok, gd_line = run_godot()
        print(gd_line)
        ok = ok and gd_ok
        ok = ok and not orphans
    now = metrics()
    if args.update:
        BASELINE.write_text(json.dumps(now, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"baseline written: {len(now)} metrics")
        return 0 if ok else 1
    if not BASELINE.exists():
        print("no baseline; run with --update")
        print(json.dumps(now, indent=2, sort_keys=True))
        return 0 if ok else 1
    drift = compare(now, json.loads(BASELINE.read_text(encoding="utf-8")))
    if drift:
        print("\n".join(drift))
        print("VERDICT: FAIL (metric drift)")
        return 1
    print(f"{len(now)} metrics within {TOLERANCE:.0%} of baseline")
    # Last line, always. Failures print above this, and reading the tail of
    # this output is exactly how two broken suites got reported as green.
    print("VERDICT: PASS" if ok else "VERDICT: FAIL (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
