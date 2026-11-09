"""Make a blind scoring sheet from a finished experiment folder.

Usage:
    python scripts/make_scoring_sheet.py traces/experiment_2026-11-09

Writes scoring/sheet.csv (shuffled; no model, hints, run index or trace file) and scoring/key.csv
(row_id -> condition). Do not open key.csv until scoring is done. Existing files are never
overwritten unless you pass --force.
"""

import argparse
import json
import random
import sys
from pathlib import Path

from agent.config import ROOT
from agent.experiment import PLAN_FILE, build_sheet, completed_entries, read_manifest, write_sheet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", type=Path, help="Experiment folder (traces/experiment_<date>)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "scoring")
    parser.add_argument("--allow-incomplete", action="store_true", help="Make a sheet even if some runs are missing")
    parser.add_argument("--force", action="store_true", help="Overwrite existing sheet.csv / key.csv")
    args = parser.parse_args()

    planned = len(json.loads((args.experiment / PLAN_FILE).read_text(encoding="utf-8"))["order"])
    completed = len(completed_entries(read_manifest(args.experiment)))
    if completed < planned and not args.allow_incomplete:
        sys.exit(f"Only {completed}/{planned} runs are complete. Finish the experiment or pass --allow-incomplete.")

    sheet, key = build_sheet(args.experiment, random.SystemRandom())
    try:
        write_sheet(args.out_dir, sheet, key, force=args.force)
    except FileExistsError as e:
        sys.exit(str(e))
    flagged = sum(bool(r["flag"]) for r in sheet)
    print(f"Wrote {len(sheet)} rows to {args.out_dir / 'sheet.csv'} ({flagged} flagged) and the key to {args.out_dir / 'key.csv'}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
