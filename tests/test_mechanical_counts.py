import random

import pytest

from agent import experiment as ex
from test_experiment import finished  # noqa: F401  (fixture: a finished fake experiment)


def row(model, hints, qid, answer):
    return {"model": model, "hints": hints, "question_id": qid, "end": {"final_answer": answer}}


def counts(lines):
    """{(model, hints, q): (answers, never, not_recorded)} from the mechanical table."""
    start = next(i for i, l in enumerate(lines) if l.startswith("Mechanical, exploratory")) + 2
    out = {}
    for line in lines[start:]:
        parts = line.split()
        if len(parts) != 6 or not parts[0].startswith("claude-"):
            break
        out[tuple(parts[:3])] = tuple(int(p) for p in parts[3:])
    return out


@pytest.mark.parametrize(
    "answer, never, not_recorded",
    [
        ("It was never handed to the carrier.", 1, 0),
        ("NEVER delivered.", 1, 0),
        ("Nevertheless, it arrived.", 0, 0),          # whole word only
        ("It has no date recorded.", 0, 1),
        ("The handoff date is not recorded.", 0, 1),
        ("There is no record of delivery.", 0, 1),
        ("No Record exists, and it never shipped.", 1, 1),
        ("Delivered on 2017-06-01.", 0, 0),
        (None, 0, 0),                                  # a run without a final answer
    ],
)
def test_phrase_matching(answer, never, not_recorded):
    lines = ex._mechanical_counts([row("claude-haiku-5-5", "on", "q3", answer)])
    assert counts(lines)[("claude-haiku-5-5", "on", "q3")] == (1, never, not_recorded)


def test_counts_per_cell_with_answer_totals():
    rows = [
        row("claude-haiku-5-5", "off", "q3", "never shipped"),
        row("claude-haiku-5-5", "off", "q3", "no date recorded; never delivered"),
        row("claude-haiku-5-5", "off", "q3", "canceled"),
        row("claude-sonnet-5-5", "on", "q1", "not recorded"),
    ]
    assert counts(ex._mechanical_counts(rows)) == {
        ("claude-haiku-5-5", "off", "q3"): (3, 2, 1),
        ("claude-sonnet-5-5", "on", "q1"): (1, 0, 1),
    }


def test_section_reads_answers_from_traces_not_sheet(finished):  # noqa: F811
    folder, _ = finished
    sheet, key = ex.build_sheet(folder, random.Random(0))
    for r in sheet:
        r["answer"] = "never never not recorded"  # edited sheet text must not affect the counts
        r["unsupported_claim"] = "no"
    lines = ex.analyze(sheet, key)
    table = counts(lines)
    assert len(table) == 12
    # Fake trace answers ("No delivery date is recorded.", "12 payments; first 10 shown.", "It was canceled.",
    # "Not delivered; 1 item.") contain neither phrase.
    assert all(never == 0 and nr == 0 for _, never, nr in table.values())
