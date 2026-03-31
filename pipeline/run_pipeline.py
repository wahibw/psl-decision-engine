# pipeline/run_pipeline.py
# Orchestrator -- runs all pipeline steps in order.
#
# Steps:
#   1. parse_cricsheet         -> ball_by_ball.parquet
#   2. build_features          -> player_stats.parquet, venue_stats.csv, matchup_matrix.parquet
#   3. build_opposition_profiles -> opposition_profiles.csv
#   4. build_partnership_history -> partnership_history.parquet
#   5. build_recent_form       -> recent_form.parquet
#
# Usage:
#   python pipeline/run_pipeline.py                   # run all steps
#   python pipeline/run_pipeline.py --step 2          # run only step 2 onwards
#   python pipeline/run_pipeline.py --only 3          # run only step 3
#   python pipeline/run_pipeline.py --incremental     # ingest new JSON only, then re-run steps 2-5
#
# Each step is skipped if its output already exists AND --force is not set.

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

PROJ_ROOT     = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJ_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# STEP DEFINITIONS
# ---------------------------------------------------------------------------

STEPS = [
    {
        "num":     1,
        "name":    "parse_cricsheet",
        "output":  PROCESSED_DIR / "ball_by_ball.parquet",
        "module":  "pipeline.parse_cricsheet",
        "fn":      "run",
    },
    {
        "num":     2,
        "name":    "build_features",
        "output":  PROCESSED_DIR / "player_stats.parquet",
        "module":  "pipeline.build_features",
        "fn":      "run",
    },
    {
        "num":     3,
        "name":    "build_opposition_profiles",
        "output":  PROCESSED_DIR / "opposition_profiles.csv",
        "module":  "pipeline.build_opposition_profiles",
        "fn":      "run",
    },
    {
        "num":     4,
        "name":    "build_partnership_history",
        "output":  PROCESSED_DIR / "partnership_history.parquet",
        "module":  "pipeline.build_partnership_history",
        "fn":      "run",
    },
    {
        "num":     5,
        "name":    "build_recent_form",
        "output":  PROCESSED_DIR / "recent_form.parquet",
        "module":  "pipeline.build_recent_form",
        "fn":      "run",
    },
]


# ---------------------------------------------------------------------------
# RUNNER
# ---------------------------------------------------------------------------

def _run_step(step: dict, force: bool, verbose: bool) -> bool:
    """
    Execute one pipeline step.
    Returns True on success, False on failure.
    """
    num    = step["num"]
    name   = step["name"]
    output = step["output"]

    if output.exists() and not force:
        size_kb = output.stat().st_size / 1024
        print(f"  [SKIP] Step {num}: {name} -- output exists ({size_kb:.0f} KB), use --force to re-run")
        return True

    print(f"\n  [RUN] Step {num}: {name}")
    t0 = time.time()

    try:
        mod = __import__(step["module"], fromlist=[step["fn"]])
        fn  = getattr(mod, step["fn"])
        fn(verbose=verbose)
        elapsed = time.time() - t0
        size_kb = output.stat().st_size / 1024 if output.exists() else 0
        print(f"  [OK]  Step {num}: {name}  ({elapsed:.1f}s, {size_kb:.0f} KB)")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [FAIL] Step {num}: {name}  ({elapsed:.1f}s)")
        print(f"         Error: {e}")
        return False


def run_pipeline(
    from_step: int  = 1,
    only_step: int | None = None,
    force:     bool = False,
    verbose:   bool = True,
) -> bool:
    """
    Run the full pipeline (or a subset).

    Args:
        from_step: Start from this step number (inclusive). Default 1.
        only_step: Run only this step number, skip all others.
        force:     Re-run even if output file exists.
        verbose:   Pass verbose=True to each step function.

    Returns:
        True if all executed steps succeeded.
    """
    t_total = time.time()
    print("=" * 55)
    print("  PSL Decision Engine -- Pipeline Run")
    print("=" * 55)

    steps_to_run = [
        s for s in STEPS
        if (only_step is None and s["num"] >= from_step)
        or (only_step is not None and s["num"] == only_step)
    ]

    if not steps_to_run:
        print("  No steps matched. Nothing to run.")
        return True

    all_ok = True
    for step in steps_to_run:
        ok = _run_step(step, force=force, verbose=verbose)
        if not ok:
            all_ok = False
            print(f"\n  Pipeline aborted at step {step['num']}: {step['name']}")
            break

    elapsed = time.time() - t_total
    print(f"\n{'='*55}")
    if all_ok:
        print(f"  Pipeline complete  [{elapsed:.1f}s total]")
    else:
        print(f"  Pipeline FAILED    [{elapsed:.1f}s total]")
    print("=" * 55)

    return all_ok


# ---------------------------------------------------------------------------
# INCREMENTAL MODE
# ---------------------------------------------------------------------------

def run_incremental(
    raw_dir: Optional[Path] = None,
    verbose: bool = True,
) -> bool:
    """
    Incrementally ingest new Cricsheet JSON files, then re-run downstream
    pipeline steps (2-5) only if new data was added.

    Args:
        raw_dir: Directory with *.json files (auto-detected if None).
        verbose: Print progress.

    Returns:
        True if everything succeeded (including if nothing was new).
    """
    from pipeline.incremental_ingest import run as ingest_run

    print("=" * 55)
    print("  PSL Decision Engine -- Incremental Ingest")
    print("=" * 55)

    try:
        result = ingest_run(raw_dir=raw_dir, verbose=verbose)
    except FileNotFoundError as exc:
        print(f"  [ERROR] {exc}")
        return False

    if not result.anything_new:
        print("\n  Database is up-to-date. No downstream rebuild needed.")
        print("=" * 55)
        return True

    # New data landed — re-run steps 2-5 with force so derived tables refresh
    print(f"\n  {result.new_matches} new match(es), {result.new_deliveries} new deliveries.")
    print(f"  Re-running downstream steps 2-5 (--force) ...\n")

    return run_pipeline(from_step=2, force=True, verbose=verbose)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PSL Decision Engine pipeline orchestrator"
    )
    p.add_argument(
        "--step", type=int, default=1, metavar="N",
        help="Run from step N onwards (default: 1 = run all)"
    )
    p.add_argument(
        "--only", type=int, default=None, metavar="N",
        help="Run only step N"
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-run steps even if output already exists"
    )
    p.add_argument(
        "--incremental", action="store_true",
        help="Ingest only new/changed JSON files, then re-run steps 2-5 if needed"
    )
    p.add_argument(
        "--dir", type=Path, default=None, metavar="PATH",
        help="Raw JSON directory (used with --incremental; default: auto-detect)"
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-step verbose output"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.incremental:
        ok = run_incremental(
            raw_dir = args.dir,
            verbose = not args.quiet,
        )
    else:
        ok = run_pipeline(
            from_step = args.step,
            only_step = args.only,
            force     = args.force,
            verbose   = not args.quiet,
        )

    sys.exit(0 if ok else 1)
