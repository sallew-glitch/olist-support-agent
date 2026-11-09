"""Compare the blind re-scoring with the original verdicts. Reads files only; never writes.

Usage:
    python scripts/compare_rescore.py              # reads scoring/rescore_sheet.csv, rescore_key.csv, sheet.csv
    python scripts/compare_rescore.py --scoring-dir path/to/scoring

Prints counts of agreements and disagreements by verdict pair, then the disagreeing rows with both
verdicts and both which_claim notes. Model and hint condition are never printed. Blank or invalid
verdicts are counted and warned about, not compared. CSV or tab-separated files are both accepted.
"""

import argparse
import sys
from pathlib import Path

from agent.config import ROOT
from agent.experiment import compare_rescore, csv_format, read_csv


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scoring-dir", type=Path, default=ROOT / "scoring")
    args = parser.parse_args(argv)

    paths = {name: args.scoring_dir / name for name in ("sheet.csv", "rescore_sheet.csv", "rescore_key.csv")}
    for name, path in paths.items():
        encoding, delimiter = csv_format(path)
        if (encoding, delimiter) != ("utf-8", ","):
            print(f"Note: {name} read as {encoding}, {'tab' if delimiter == chr(9) else 'comma'}-separated.")
    lines = compare_rescore(read_csv(paths["sheet.csv"]), read_csv(paths["rescore_sheet.csv"]), read_csv(paths["rescore_key.csv"]))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
