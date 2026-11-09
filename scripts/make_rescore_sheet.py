"""Make a blind re-scoring sheet from the scored sheet: every yes/unclear row plus N random "no" rows.

Usage:
    python scripts/make_rescore_sheet.py              # 12 "no" rows (default)
    python scripts/make_rescore_sheet.py --n-no 12

Writes scoring/rescore_sheet.csv (verdicts cleared, new row_ids, shuffled) and scoring/rescore_key.csv
(new row_id -> original row_id and condition). Prints no verdicts. Never overwrites unless --force.
"""

import argparse
import random
import sys
from pathlib import Path

from agent.config import ROOT
from agent.experiment import RESCORE_KEY_COLUMNS, build_rescore, read_csv, write_sheet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scoring-dir", type=Path, default=ROOT / "scoring")
    parser.add_argument("--n-no", type=int, default=12, help='How many random "no" rows to include')
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    sheet = read_csv(args.scoring_dir / "sheet.csv")
    key = read_csv(args.scoring_dir / "key.csv")
    rescore, rescore_key = build_rescore(sheet, key, args.n_no, random.SystemRandom())
    try:
        write_sheet(args.scoring_dir, rescore, rescore_key, force=args.force,
                    names=("rescore_sheet.csv", "rescore_key.csv"), key_columns=RESCORE_KEY_COLUMNS)
    except FileExistsError as e:
        sys.exit(str(e))
    print(f"Wrote {len(rescore)} rows to {args.scoring_dir / 'rescore_sheet.csv'} and the key to "
          f"{args.scoring_dir / 'rescore_key.csv'}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
