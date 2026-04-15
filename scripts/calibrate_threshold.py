"""Run all baselines locally and suggest a REGISTRATION_THRESHOLD value."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_local(baseline: str, weights: Path | None) -> dict:
    cmd = [sys.executable, str(ROOT / "scripts" / "run_local_eval.py"), "--baseline", baseline]
    if weights is not None:
        cmd.extend(["--weights", str(weights)])
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # Parse last JSON blob from stdout.
    text = out.stdout.strip()
    idx = text.rfind("{\n")
    return json.loads(text[idx:])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trivial-weights", default=None)
    p.add_argument("--domain-weights", default=None)
    args = p.parse_args()

    results = {}
    for name, w in [("degenerate", None), ("trivial", args.trivial_weights), ("domain_adapted", args.domain_weights)]:
        print(f"--- running {name} ---")
        try:
            results[name] = run_local(name, Path(w) if w else None)
        except subprocess.CalledProcessError as exc:
            print(f"baseline {name} failed: {exc.stderr}", file=sys.stderr)

    for name, res in results.items():
        d = res["details"]
        print(
            f"{name}: final={res['final']:.4f} reg={d['registration_accuracy']:.4f} int={d['integration_score']:.4f}"
        )

    # Suggestion: set threshold between degenerate.reg and trivial.reg (midpoint), clipped into [0.2, 0.5].
    if "degenerate" in results and "trivial" in results:
        deg = results["degenerate"]["details"]["registration_accuracy"]
        triv = results["trivial"]["details"]["registration_accuracy"]
        midpoint = (deg + triv) / 2
        suggested = max(0.2, min(0.5, midpoint))
        print(f"suggested REGISTRATION_THRESHOLD: {suggested:.3f} (degenerate={deg:.3f}, trivial={triv:.3f})")


if __name__ == "__main__":
    main()
