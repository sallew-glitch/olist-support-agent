"""Build data/olist.db from the Olist Kaggle download.

Usage:
    python scripts/prepare_data.py --zip path/to/archive.zip
    python scripts/prepare_data.py --csv-dir data/raw/archive   # already extracted

With --zip, the CSV files are extracted into data/raw/ first. The database is
built in a temporary file and moved into place only when complete.
"""

import argparse
import os
import sqlite3
import sys
import zipfile
from pathlib import Path

import pandas as pd

from agent.schema import TABLES, create_schema

ROOT = Path(__file__).resolve().parent.parent


def extract_csvs(zip_path: Path, dest: Path) -> None:
    """Extract only .csv members, flattened to their base names (no paths from the zip are trusted)."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            name = Path(member.filename).name
            if member.is_dir() or not name.lower().endswith(".csv"):
                continue
            with zf.open(member) as src, open(dest / name, "wb") as out:
                out.write(src.read())
            print(f"extracted {name}")


def load_table(conn: sqlite3.Connection, table: str, spec: dict, csv_dir: Path) -> None:
    path = csv_dir / spec["csv"]
    if not path.exists():
        sys.exit(f"Missing file: {path}")

    # Read everything as text first; utf-8-sig strips the BOM some files have.
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8-sig")

    kept = [name for name, _ in spec["columns"]]
    expected = set(kept) | set(spec.get("dropped", []))
    if set(df.columns) != expected:
        sys.exit(
            f"{path.name}: unexpected columns.\n"
            f"  missing: {sorted(expected - set(df.columns))}\n"
            f"  extra:   {sorted(set(df.columns) - expected)}"
        )

    df = df[kept]
    for name, sql_type in spec["columns"]:
        if sql_type == "INTEGER":
            df[name] = pd.to_numeric(df[name]).astype("Int64")
        elif sql_type == "REAL":
            df[name] = pd.to_numeric(df[name])

    # Convert pandas missing values to None so SQLite stores NULL.
    rows = df.astype(object).where(df.notna(), None).itertuples(index=False, name=None)
    placeholders = ", ".join("?" for _ in kept)
    conn.executemany(f"INSERT INTO {table} ({', '.join(kept)}) VALUES ({placeholders})", rows)
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--zip", type=Path, help="Path to the Kaggle zip file")
    source.add_argument("--csv-dir", type=Path, help="Folder that already contains the extracted CSVs")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw", help="Where --zip extracts to")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "olist.db", help="Output database path")
    args = parser.parse_args()

    if args.zip:
        if not args.zip.exists():
            sys.exit(f"Zip not found: {args.zip}")
        extract_csvs(args.zip, args.raw_dir)
        csv_dir = args.raw_dir
    else:
        csv_dir = args.csv_dir

    args.db.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.db.with_suffix(".db.tmp")
    tmp_path.unlink(missing_ok=True)

    conn = sqlite3.connect(tmp_path)
    try:
        create_schema(conn)
        for table, spec in TABLES.items():
            load_table(conn, table, spec, csv_dir)
        counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}
    finally:
        conn.close()
    os.replace(tmp_path, args.db)

    print(f"\nBuilt {args.db}")
    for table, count in counts.items():
        print(f"  {table:<36} {count:>8,}")
    print("(geolocation is not loaded; review comment text is deliberately not loaded)")


if __name__ == "__main__":
    main()
