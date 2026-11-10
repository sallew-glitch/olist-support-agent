"""Join the scored sheet with the key and print raw counts. No percentages, no tests, no conclusions.

Usage:
    python scripts/analyze_experiment.py              # reads scoring/sheet.csv, key.csv and corrections.csv
    python scripts/analyze_experiment.py --scoring-dir path/to/scoring

The as-scored tables come first, unchanged. If corrections.csv exists, "after corrections" tables
follow at the end; sheet.csv itself is never modified.
"""

import argparse
import sys
from pathlib import Path

from agent.config import ROOT
from agent.experiment import analyze, csv_format, read_csv


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scoring-dir", type=Path, default=ROOT / "scoring")
    args = parser.parse_args(argv)
    encoding, delimiter = csv_format(args.scoring_dir / "sheet.csv")
    if (encoding, delimiter) != ("utf-8", ","):
        print(f"Note: sheet.csv was read as {encoding}, {'tab' if delimiter == chr(9) else 'comma'}-separated "
              "(re-saved by a spreadsheet app). Answer text is taken from the traces, not the sheet.")
    sheet = read_csv(args.scoring_dir / "sheet.csv")
    key = read_csv(args.scoring_dir / "key.csv")
    corrections_path = args.scoring_dir / "corrections.csv"
    corrections = read_csv(corrections_path) if corrections_path.exists() else None
    print("\n".join(analyze(sheet, key, corrections=corrections)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
