import hashlib
import importlib.util
import random

import pytest

from agent import experiment as ex
from agent.config import ROOT
from test_experiment import finished  # noqa: F401  (fixture: a finished fake experiment)

spec = importlib.util.spec_from_file_location("analyze_experiment", ROOT / "scripts" / "analyze_experiment.py")
analyze_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze_script)

AFTER = "Totals per model x hints (counts out of N) - after corrections"
BEFORE = "Totals per model x hints (counts out of N)"


@pytest.fixture
def scored(finished):  # noqa: F811
    """All 12 rows scored "no"."""
    folder, _ = finished
    sheet, key = ex.build_sheet(folder, random.Random(0))
    for row in sheet:
        row["unsupported_claim"] = "no"
    return sheet, key


def totals(lines, title):
    """{(model, hints): (N, yes, no, unclear)} from a totals table."""
    start = lines.index(title) + 2
    out = {}
    for line in lines[start:start + 4]:
        model, hints, n, yes, no, unclear, *_ = line.split()
        out[(model, hints)] = (int(n), int(yes), int(no), int(unclear))
    return out


def correction(row_id, original="no", corrected="yes", claim="some claim"):
    return {"row_id": row_id, "original_verdict": original, "corrected_verdict": corrected,
            "which_claim": claim, "found_by": "second-reader audit blind to condition"}


def test_as_scored_output_is_unchanged(scored):
    sheet, key = scored
    plain = ex.analyze(sheet, key)
    with_corrections = ex.analyze(sheet, key, corrections=[correction(sheet[0]["row_id"])])
    assert with_corrections[:len(plain)] == plain  # the as-scored part is printed exactly as before
    assert ex.analyze(sheet, key, corrections=None) == plain


def test_corrections_change_only_the_after_tables(scored):
    sheet, key = scored
    key_by_id = {k["row_id"]: k for k in key}
    target = sheet[0]["row_id"]
    condition = (key_by_id[target]["model"], key_by_id[target]["hints"])
    lines = ex.analyze(sheet, key, corrections=[correction(target, corrected="unclear")])

    before, after = totals(lines, BEFORE), totals(lines, AFTER)
    assert before[condition] == (3, 0, 3, 0)
    assert after[condition] == (3, 0, 2, 1)
    assert {c: v for c, v in after.items() if c != condition} == {c: v for c, v in before.items() if c != condition}
    marked = [l for l in lines if "[CORRECTED]" in l]
    assert len(marked) == 1 and marked[0].startswith(f"- {target} [CORRECTED]") and "no -> unclear" in marked[0]


@pytest.mark.parametrize(
    "make, warning",
    [
        (lambda s: [correction(s[0]["row_id"], original="yes")], "corrections say original was yes"),
        (lambda s: [correction("R0000")], "unknown row"),
        (lambda s: [correction(s[0]["row_id"]), correction(s[0]["row_id"])], "appears 2 times"),
        (lambda s: [correction(s[0]["row_id"], corrected="maybe")], "is not yes/no/unclear"),
    ],
)
def test_bad_corrections_are_warned_and_not_applied(scored, make, warning):
    sheet, key = scored
    lines = ex.analyze(sheet, key, corrections=make(sheet))
    assert any(warning in l for l in lines)
    assert not any("[CORRECTED]" in l for l in lines)
    assert totals(lines, AFTER) == totals(lines, BEFORE)


def test_script_reads_corrections_and_writes_nothing(scored, tmp_path, capsys):
    sheet, key = scored
    d = tmp_path / "scoring"
    ex.write_sheet(d, sheet, key)
    ex.write_csv(d / "corrections.csv", ["row_id", "original_verdict", "corrected_verdict", "which_claim", "found_by"],
                 [correction(sheet[1]["row_id"], claim="a, quoted claim")])
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir()}

    assert analyze_script.main(["--scoring-dir", str(d)]) == 0
    out = capsys.readouterr().out
    assert "AFTER CORRECTIONS" in out and f"- {sheet[1]['row_id']} [CORRECTED]" in out and "a, quoted claim" in out
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir()} == before


def test_script_without_corrections_file(scored, tmp_path, capsys):
    sheet, key = scored
    ex.write_sheet(tmp_path, sheet, key)
    analyze_script.main(["--scoring-dir", str(tmp_path)])
    assert "AFTER CORRECTIONS" not in capsys.readouterr().out
