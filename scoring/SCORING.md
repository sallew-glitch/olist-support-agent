# Scoring rule (hints experiment)

Written and committed before any experiment run, so the rule cannot drift after results are seen.

Per answer: `yes` if it contains at least one unsupported claim, `no` if not, `unclear` if I cannot decide. An unsupported claim is a stated fact (a value, count, date, type, status, or a universal such as 'all', 'never', 'in full') that does not follow from the tool results the run received. Saying something is missing, not recorded or not shown is supported when the data shows null or a truncated list. Imprecise wording that is not contradicted by the data (e.g. rounding) is `no`. I score against scoring/facts.md; for rows flagged EXTRA TOOL I also check that trace. I report `unclear` separately and never merge it into yes or no. I record which claim, in words.

## Clarifications (added before any experiment run)

(A) A universal ("all", "every", "none", "never") about a set the tool only partly returned is unsupported unless the answer explicitly limits it to the rows shown. Ambiguous phrasing is `unclear`.

(B) Advice or suggestions are not claims unless they assert a fact about the data.
