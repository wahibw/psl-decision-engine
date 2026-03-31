# pipeline/incremental_ingest.py
# Incremental Cricsheet ingestion — only parses NEW or CHANGED match files.
#
# How it works:
#   1. Reads data/processed/ingest_manifest.json  (created on first run)
#      Manifest records {filename: {mtime, size, parsed_at}} for every file
#      that has already been ingested into ball_by_ball.parquet.
#   2. Scans the raw JSON directory for files NOT in the manifest (or whose
#      mtime/size has changed since last parse).
#   3. Parses only those new/changed files using parse_cricsheet._parse_match().
#   4. Merges the new rows into the existing ball_by_ball.parquet, deduplicates
#      on (match_id, innings, over, ball), and writes the result back.
#   5. Updates the manifest so the next run skips these files.
#
# Returns:
#   IngestResult(new_matches, new_deliveries, updated_matches, skipped, errors)
#
# Usage:
#   python pipeline/incremental_ingest.py          # default raw dir
#   python pipeline/incremental_ingest.py --dir /path/to/json_files
#   from pipeline.incremental_ingest import run    # called by run_pipeline --incremental

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import pandas as pd

# Re-use the proven single-match parser and alias loader from parse_cricsheet
from pipeline.parse_cricsheet import (
    _parse_match,
    _load_alias_map,
    PLAYER_INDEX_SPEC,
    PLAYER_INDEX_ACTUAL,
    VENUE_FIX,
    RAW_FALLBACK,
)

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

PROJ_ROOT      = Path(__file__).resolve().parent.parent
RAW_DIR        = PROJ_ROOT / "data" / "raw"
PROCESSED_DIR  = PROJ_ROOT / "data" / "processed"
BBB_FILE       = PROCESSED_DIR / "ball_by_ball.parquet"
MANIFEST_FILE  = PROCESSED_DIR / "ingest_manifest.json"


# ---------------------------------------------------------------------------
# RESULT TYPE
# ---------------------------------------------------------------------------

@dataclass
class IngestResult:
    """
    Summary of one incremental ingest run.

    Attributes:
        new_matches:      Files parsed for the first time.
        updated_matches:  Files re-parsed because mtime/size changed.
        new_deliveries:   Net new delivery rows added to ball_by_ball.parquet.
        skipped:          Files already up-to-date in the manifest (not re-parsed).
        errors:           Files that failed to parse (malformed JSON etc.).
        raw_dir:          Directory that was scanned.
        anything_new:     True if the parquet was updated (downstream steps needed).
    """
    new_matches:     int  = 0
    updated_matches: int  = 0
    new_deliveries:  int  = 0
    skipped:         int  = 0
    errors:          int  = 0
    raw_dir:         str  = ""
    anything_new:    bool = False


# ---------------------------------------------------------------------------
# MANIFEST HELPERS
# ---------------------------------------------------------------------------

def _load_manifest() -> dict[str, dict]:
    """Load the existing manifest, or return empty dict on first run."""
    if not MANIFEST_FILE.exists():
        return {}
    try:
        with open(MANIFEST_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_manifest(manifest: dict[str, dict]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _file_fingerprint(fp: Path) -> dict:
    """Return a dict with mtime (float) and size (bytes) for change detection."""
    stat = fp.stat()
    return {"mtime": stat.st_mtime, "size": stat.st_size}


def _needs_parse(fp: Path, manifest: dict[str, dict]) -> bool:
    """Return True if fp is new or has changed since the last manifest entry."""
    key = fp.name
    if key not in manifest:
        return True
    recorded = manifest[key]
    current  = _file_fingerprint(fp)
    return current["mtime"] != recorded.get("mtime") or current["size"] != recorded.get("size")


# ---------------------------------------------------------------------------
# ALIAS MAP RESOLUTION (same logic as parse_cricsheet.run())
# ---------------------------------------------------------------------------

def _resolve_alias_map() -> tuple[dict[str, str], set[str]]:
    if PLAYER_INDEX_SPEC.exists():
        return _load_alias_map(PLAYER_INDEX_SPEC)
    if PLAYER_INDEX_ACTUAL.exists():
        return _load_alias_map(PLAYER_INDEX_ACTUAL)
    return {}, set()


# ---------------------------------------------------------------------------
# CORE INCREMENTAL LOGIC
# ---------------------------------------------------------------------------

def run(
    raw_dir:    Optional[Path] = None,
    verbose:    bool = True,
) -> IngestResult:
    """
    Incrementally ingest new/changed Cricsheet JSON files into ball_by_ball.parquet.

    Args:
        raw_dir:  Directory containing *.json match files.
                  Falls back to data/raw/ then ../psl_json/ (same as parse_cricsheet).
        verbose:  Print progress.

    Returns:
        IngestResult summarising what changed.
    """
    t0 = time.time()

    # ── Resolve raw dir ──────────────────────────────────────────────────────
    if raw_dir is None:
        if list(RAW_DIR.glob("*.json")):
            raw_dir = RAW_DIR
        elif RAW_FALLBACK.exists() and list(RAW_FALLBACK.glob("*.json")):
            raw_dir = RAW_FALLBACK
        else:
            raise FileNotFoundError(
                f"No JSON files found in {RAW_DIR} or {RAW_FALLBACK}. "
                "Place Cricsheet PSL JSON files in data/raw/ and re-run."
            )

    result = IngestResult(raw_dir=str(raw_dir))
    json_files = sorted(raw_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No *.json files found in {raw_dir}")

    if verbose:
        print(f"[incremental_ingest] Scanning {len(json_files)} files in {raw_dir}")

    # ── Load manifest & identify new/changed files ───────────────────────────
    manifest = _load_manifest()
    to_parse  = [fp for fp in json_files if _needs_parse(fp, manifest)]
    to_skip   = len(json_files) - len(to_parse)
    result.skipped = to_skip

    if not to_parse:
        if verbose:
            print(f"[incremental_ingest] All {len(json_files)} files already ingested — nothing to do.")
        return result

    if verbose:
        print(f"[incremental_ingest] {len(to_parse)} new/changed file(s) to parse "
              f"({to_skip} already up-to-date)")

    # ── Load alias map ───────────────────────────────────────────────────────
    alias, _ = _resolve_alias_map()

    # ── Parse new files ──────────────────────────────────────────────────────
    new_rows: list[dict] = []
    updated_match_ids:  set[str] = set()
    newly_parsed_files: list[Path] = []

    for fp in to_parse:
        is_update = fp.name in manifest   # previously parsed but file changed
        rows = _parse_match(fp, alias)
        if rows:
            new_rows.extend(rows)
            newly_parsed_files.append(fp)
            if is_update:
                result.updated_matches += 1
                updated_match_ids.add(fp.stem)
            else:
                result.new_matches += 1
        else:
            result.errors += 1
            if verbose:
                print(f"  [WARN] Could not parse {fp.name} — skipped")

    if not new_rows:
        if verbose:
            print("[incremental_ingest] Parsed files yielded no rows — parquet unchanged.")
        # Still update manifest for files we attempted (even if empty/errored)
        _update_manifest_entries(manifest, newly_parsed_files)
        return result

    new_df = pd.DataFrame(new_rows)

    # ── Load existing parquet (if it exists) ─────────────────────────────────
    if BBB_FILE.exists():
        existing = pd.read_parquet(BBB_FILE)
        rows_before = len(existing)

        # For updated (changed) files: drop old rows for those match_ids first
        if updated_match_ids:
            existing = existing[~existing["match_id"].isin(updated_match_ids)]
            if verbose:
                dropped = rows_before - len(existing)
                print(f"[incremental_ingest] Dropped {dropped} stale rows for "
                      f"{len(updated_match_ids)} updated match(es)")

        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        if verbose:
            print("[incremental_ingest] ball_by_ball.parquet not found — creating fresh.")
        combined = new_df
        rows_before = 0

    # ── Deduplicate (same logic as parse_cricsheet) ──────────────────────────
    key_cols = ["match_id", "innings", "over", "ball"]
    dups = combined.duplicated(subset=key_cols, keep="first")
    if dups.sum() > 0:
        combined = combined[~dups].reset_index(drop=True)
        if verbose:
            print(f"[incremental_ingest] Removed {dups.sum()} duplicate deliveries")

    # ── Enforce dtypes ────────────────────────────────────────────────────────
    for col, dtype in [
        ("season",    "int16"),
        ("over",      "int8"),
        ("innings",   "int8"),
        ("is_wide",   bool),
        ("is_noball", bool),
        ("is_wicket", bool),
    ]:
        if col in combined.columns:
            combined[col] = combined[col].astype(dtype)
    if "phase" in combined.columns:
        combined["phase"] = combined["phase"].astype("category")

    # ── Sort ──────────────────────────────────────────────────────────────────
    combined = combined.sort_values(
        ["season", "date", "match_id", "innings", "over", "ball"]
    ).reset_index(drop=True)

    # ── Write ─────────────────────────────────────────────────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(BBB_FILE, index=False)

    result.new_deliveries = len(combined) - rows_before
    result.anything_new   = True

    elapsed = time.time() - t0
    if verbose:
        size_kb = BBB_FILE.stat().st_size / 1024
        total_matches = combined["match_id"].nunique()
        print(
            f"[incremental_ingest] Done: +{result.new_matches} new match(es), "
            f"+{result.updated_matches} updated, "
            f"+{result.new_deliveries} new deliveries  "
            f"| {total_matches} total matches, {len(combined):,} rows "
            f"| {size_kb:.0f} KB  [{elapsed:.1f}s]"
        )

    # ── Update manifest ───────────────────────────────────────────────────────
    _update_manifest_entries(manifest, newly_parsed_files)

    return result


def _update_manifest_entries(manifest: dict[str, dict], files: list[Path]) -> None:
    """Record fingerprint + timestamp for each successfully parsed file."""
    now = time.time()
    for fp in files:
        manifest[fp.name] = {
            **_file_fingerprint(fp),
            "parsed_at": now,
        }
    _save_manifest(manifest)


# ---------------------------------------------------------------------------
# STATUS REPORT
# ---------------------------------------------------------------------------

def seed_manifest(raw_dir: Optional[Path] = None, verbose: bool = True) -> int:
    """
    Bootstrap the manifest by fingerprinting every JSON file in raw_dir
    WITHOUT re-parsing them. Use this once after data/raw already has all
    historical files and ball_by_ball.parquet is already built, so that
    future incremental runs only process truly new files.

    Returns the number of files registered.
    """
    if raw_dir is None:
        if list(RAW_DIR.glob("*.json")):
            raw_dir = RAW_DIR
        elif RAW_FALLBACK.exists() and list(RAW_FALLBACK.glob("*.json")):
            raw_dir = RAW_FALLBACK
        else:
            print("[seed_manifest] No raw data directory found.")
            return 0

    json_files = sorted(raw_dir.glob("*.json"))
    if not json_files:
        print(f"[seed_manifest] No *.json files found in {raw_dir}")
        return 0

    manifest = _load_manifest()
    now = time.time()
    added = 0
    for fp in json_files:
        if fp.name not in manifest:
            manifest[fp.name] = {
                **_file_fingerprint(fp),
                "parsed_at": now,
            }
            added += 1

    _save_manifest(manifest)
    if verbose:
        print(f"[seed_manifest] Registered {added} file(s) in manifest "
              f"({len(manifest)} total). Future --incremental runs will skip these.")
    return added


def status(raw_dir: Optional[Path] = None) -> None:
    """
    Print a human-readable report of manifest state vs. raw dir.
    Does NOT parse anything.
    """
    if raw_dir is None:
        if list(RAW_DIR.glob("*.json")):
            raw_dir = RAW_DIR
        elif RAW_FALLBACK.exists():
            raw_dir = RAW_FALLBACK
        else:
            print("[status] No raw data directory found.")
            return

    json_files = sorted(raw_dir.glob("*.json"))
    manifest   = _load_manifest()

    ingested   = [f for f in json_files if not _needs_parse(f, manifest)]
    pending    = [f for f in json_files if _needs_parse(f, manifest)]

    print(f"\n{'='*60}")
    print(f"  Incremental Ingest Status")
    print(f"{'='*60}")
    print(f"  Raw dir       : {raw_dir}")
    print(f"  Total files   : {len(json_files)}")
    print(f"  Up-to-date    : {len(ingested)}")
    print(f"  Pending parse : {len(pending)}")
    if BBB_FILE.exists():
        size_kb = BBB_FILE.stat().st_size / 1024
        df = pd.read_parquet(BBB_FILE, columns=["match_id", "season"])
        print(f"  ball_by_ball  : {df['match_id'].nunique()} matches, "
              f"{len(df):,} deliveries, {size_kb:.0f} KB")
        seasons = sorted(df["season"].unique())
        print(f"  Seasons in DB : {seasons}")
    else:
        print(f"  ball_by_ball  : not yet created")
    print(f"{'='*60}\n")
    if pending:
        print(f"  Files pending ingestion ({min(len(pending), 10)} shown):")
        for fp in pending[:10]:
            print(f"    {fp.name}")
        if len(pending) > 10:
            print(f"    ... and {len(pending) - 10} more")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Incremental Cricsheet JSON ingestion into ball_by_ball.parquet"
    )
    p.add_argument(
        "--dir", type=Path, default=None, metavar="PATH",
        help="Directory containing Cricsheet *.json files (default: auto-detect)"
    )
    p.add_argument(
        "--status", action="store_true",
        help="Print ingestion status report only (no parsing)"
    )
    p.add_argument(
        "--seed", action="store_true",
        help="Bootstrap manifest from existing files without re-parsing (use once after initial full pipeline run)"
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )
    return p.parse_args()


if __name__ == "__main__":
    sys.path.insert(0, str(PROJ_ROOT))
    args = _parse_args()

    if args.status:
        status(args.dir)
    elif args.seed:
        seed_manifest(args.dir, verbose=not args.quiet)
    else:
        result = run(raw_dir=args.dir, verbose=not args.quiet)
        if result.anything_new:
            print(f"\n  Downstream pipeline steps (2-5) should now be re-run.")
            print(f"  Run:  python pipeline/run_pipeline.py --incremental")
        sys.exit(0 if result.errors == 0 else 1)
