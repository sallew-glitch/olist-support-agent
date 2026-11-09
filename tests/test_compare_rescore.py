import csv
import hashlib
import importlib.util
import random

import pytest

from agent import experiment as ex
from agent.config import ROOT
from test_experiment import finished  # noqa: F401  (fixture: a finished fake experiment)

spec = importlib.util.spec_from_file_location("compare_rescore", ROOT / "scripts" / "compare_rescore.py")
compare_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_script)


def write_excel_style(path, columns, rows):
    """As Excel on Windows may save it: Windows-1252, tab-separated."""
    with open(path, "w", encoding="cp1252", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def scored(finished, tmp_path):  # noqa: F811
    """An original scored sheet, and a re-score sheet with known agreements and disagreements."""
    folder, _ = finished
    sheet, key = ex.build_sheet(folder, random.Random(0))
    originals = ["yes", "unclear", "no", "no", "no", "no", "no", "no", "no", "no", "no", "no"]
    for row, verdict in zip(sheet, originals):
        row["unsupported_claim"], row["which_claim"] = verdict, f"orig note {row['row_id']}" if verdict != "no" else ""
    rescore, rescore_key = ex.build_rescore(sheet, key, n_no=4, rng=random.Random(2))
    return sheet, key, rescore, rescore_key


def fill(rescore, rescore_key, sheet, plan):
    """Set each re-score verdict from plan[original verdict] (a list consumed in row order)."""
    orig = {r["row_id"]: r["unsupported_claim"] for r in sheet}
    to_orig = {k["row_id"]: k["original_row_id"] for k in rescore_key}
    for row in sorted(rescore, key=lambda r: r["row_id"]):
        row["unsupported_claim"] = plan[orig[to_orig[row["row_id"]]]].pop(0)
        row["which_claim"] = f"new note {row['row_id']}" if row["unsupported_claim"] in ("yes", "unclear") else ""


def test_counts_pairs_and_lists_disagreements(scored):
    sheet, key, rescore, rescore_key = scored
    fill(rescore, rescore_key, sheet, {"yes": ["no"], "unclear": ["unclear"], "no": ["no", "no", "yes", ""]})
    lines = ex.compare_rescore(sheet, rescore, rescore_key)
    report = "\n".join(lines)
    assert "Rows compared: 6 | agree: 3 | disagree: 2 | not comparable (blank/invalid): 1" in report
    table_start = lines.index("Counts by verdict pair") + 2  # skip the column header
    table = lines[table_start:lines.index("", table_start)]
    pairs = {tuple(l.split()[:2]): l.split()[2] for l in table}
    assert pairs == {("yes", "no"): "1", ("unclear", "unclear"): "1", ("no", "no"): "2", ("no", "yes"): "1", ("no", "blank"): "1"}
    assert "WARNING: blank or invalid rescore verdicts" in report
    # Disagreeing rows show both verdicts and both notes.
    disagreeing = lines[lines.index("Disagreeing rows") + 1:lines.index("Not comparable rows") - 1]
    assert len(disagreeing) == 2 * 3
    assert any("original=yes rescore=no" in l for l in disagreeing)
    assert any("original which_claim: orig note" in l for l in disagreeing)
    assert any("rescore which_claim:  new note" in l for l in disagreeing)


def test_invalid_verdicts_are_warned_not_merged(scored):
    sheet, key, rescore, rescore_key = scored
    fill(rescore, rescore_key, sheet, {"yes": ["maybe"], "unclear": ["unclear"], "no": ["no", "no", "no", "no"]})
    report = "\n".join(ex.compare_rescore(sheet, rescore, rescore_key))
    assert "=invalid" in report and "not comparable (blank/invalid): 1" in report


def test_unmatched_rows_are_warned(scored):
    sheet, key, rescore, rescore_key = scored
    fill(rescore, rescore_key, sheet, {"yes": ["yes"], "unclear": ["unclear"], "no": ["no"] * 4})
    report = "\n".join(ex.compare_rescore(sheet, rescore[1:], rescore_key))
    assert "in rescore_key.csv but not in rescore_sheet.csv" in report
    assert "Rows compared: 5" in report


def test_script_reads_excel_files_prints_no_condition_and_writes_nothing(scored, tmp_path, capsys):
    sheet, key, rescore, rescore_key = scored
    fill(rescore, rescore_key, sheet, {"yes": ["no"], "unclear": ["yes"], "no": ["no", "no", "no", "no"]})
    for rows in (sheet, rescore):  # a non-ASCII character, so the cp1252 files are not also valid UTF-8
        rows[0]["answer"] += " – São Paulo"
    d = tmp_path / "scoring"
    d.mkdir()
    write_excel_style(d / "sheet.csv", ex.SHEET_COLUMNS, sheet)
    write_excel_style(d / "rescore_sheet.csv", ex.SHEET_COLUMNS, rescore)
    ex.write_csv(d / "rescore_key.csv", ex.RESCORE_KEY_COLUMNS, rescore_key)
    ex.write_csv(d / "key.csv", ex.KEY_COLUMNS, key)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir()}

    assert compare_script.main(["--scoring-dir", str(d)]) == 0
    out = capsys.readouterr().out

    assert "Rows compared: 6 | agree: 4 | disagree: 2" in out
    assert "read as cp1252, tab-separated" in out
    for leak in ("haiku", "sonnet", "hints", ".jsonl", "|r"):
        assert leak not in out.lower(), leak
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir()}
    assert after == before  # no file changed, none added
