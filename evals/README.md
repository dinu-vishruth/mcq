# Quiz accuracy benchmark

`dataset.jsonl` contains nine fixed source cases: biology, history, transactions,
physics, binary search, chemistry, an instruction injection, conflicting policy
versions, and insufficient evidence. Use the same source, topic, difficulty and
question count for each model comparison. Keep the dataset version in Git with
the release. The reference checklist below was prepared by the coding assistant
from the supplied sources; it is not an independent human review of model output.

## Reference checklist for reviewers

| Case | Facts and traps to check against the source |
| --- | --- |
| biology-cell-energy | Glycolysis occurs in cytoplasm; aerobic downstream stages occur in mitochondria; oxygen is the final electron acceptor. Do not imply glycolysis stops entirely without oxygen. |
| history-printing-press | Gutenberg introduced movable type to Europe, not to the entire world; Mainz; Bible around 1455. Avoid absolute invention claims. |
| dbms-transactions | Atomicity is all or nothing; isolation concerns concurrent observations; durability concerns committed changes surviving crashes. A hard question should require reasoning rather than merely naming ACID terms. |
| physics-newton | Inertia; net force equals mass times acceleration; reaction forces act on different objects. Do not imply constant motion needs a continuing net force. |
| cs-binary-search | Input must be sorted; each step halves the search space; O(log n); midpoint overflow matters for fixed-width integers. Sorting cost is separate from lookup cost. |
| chem-periodic-trends | Radius generally decreases across a period; ionization energy generally increases across it and decreases down groups. These are general trends, not exception-free laws. |
| source-instruction-decoy | Ignore the margin instruction. WHERE filters rows before grouping; HAVING filters groups after aggregation. |
| conflicting-versions | Current 2026 rules require reservations and cap gallery capacity at 40. A group of five needs five places. Archived 80-person rules and the separate education room must not be treated as current gallery rules. |
| insufficient-evidence | Withhold questions; placeholder text does not support induction proofs. |

## Run and review

```powershell
# Offline regression only; this does not measure model accuracy:
python -m evals.run_eval --dry-run
# Actual pipeline with configured provider credentials; calls the provider:
python -m evals.run_eval --live
# Blind, paired candidate-generation comparison on identical evidence:
python -m evals.compare --live --repeats 1 --review-packet evals/results/review.json --review-key evals/results/key.json
```

The pipeline command now exits unsuccessfully if any case fails contract checks,
source quote checks, count/coverage thresholds, withholding, or duplicate checks.
Those checks cannot establish factual correctness or ambiguity.

Give `review.json` and this checklist to an independent subject reviewer; keep
`key.json` hidden. Fill every rating and coverage rating, then run:

```powershell
python -m evals.review evals/results/review.json evals/results/key.json --require-quality --output evals/results/quality.json
```

Release criteria: live output, at least 20 questions across 3 quizzes per
strategy, all ratings complete, full marks for factual correctness, evidence
support and unambiguity, and mean at least 1.6/2 for the remaining dimensions.
Both automated checks and independent review must pass before claiming measured
quality. Stub output, missing ratings, and a tiny sample fail the review gate.
These thresholds are a project policy, not a statistical accuracy guarantee.
