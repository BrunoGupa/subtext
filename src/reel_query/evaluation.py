"""Measurement: recall@k, answer faithfulness, and the chunk-strategy x top-k sweep.

The distinction this module exists to make visible:

* **retrieval failure** — the line that answers the question never came back at all.
  No prompt change fixes that; it is a chunking, embedding or k problem.
* **generation failure** — the line *did* come back and the answer ignored it, or
  cited something that was never retrieved. That is a prompt or model problem.

recall@k measures the first. Faithfulness measures the second. Reporting a single
"accuracy" number would hide which one you have.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import EVALS_DIR
from .retrieval import search

GOLDEN_PATH = EVALS_DIR / "golden_set.jsonl"
_LINE_ID = re.compile(r"\bline[ _]?(?:id)?[ #:]*(\d{1,6})\b", re.IGNORECASE)


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    question: str
    kind: str                      # "lookup" | "semantic" | "empty"
    expected_line_ids: list[int]
    character: str | None = None
    season: int | None = None
    notes: str = ""

    @property
    def is_empty_case(self) -> bool:
        return self.kind == "empty" or not self.expected_line_ids


def load_golden(path: Path | None = None) -> list[GoldenQuestion]:
    target = path or GOLDEN_PATH
    if not target.exists():
        raise FileNotFoundError(
            f"{target} not found. The golden set is hand-labelled; see evals/README.md."
        )
    questions: list[GoldenQuestion] = []
    for raw in target.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        record = json.loads(raw)
        questions.append(
            GoldenQuestion(
                id=record["id"],
                question=record["question"],
                kind=record.get("kind", "semantic"),
                expected_line_ids=[int(i) for i in record.get("expected_line_ids", [])],
                character=record.get("character") or None,
                season=record.get("season"),
                notes=record.get("notes", ""),
            )
        )
    return questions


def parse_strategy(label: str) -> tuple[str, int]:
    """`"line"` -> `("line", 1)`; `"window3"` -> `("window", 3)`."""
    if label == "line":
        return "line", 1
    match = re.fullmatch(r"window(\d+)", label)
    if not match:
        raise ValueError(f"unknown strategy label {label!r}; expected 'line' or 'windowN'")
    return "window", int(match.group(1))


# --------------------------------------------------------------------------- recall


# Distance thresholds the abstention curve is reported at. A brute-force vector search
# always returns k rows, so "did it correctly return nothing?" is only a question you
# can ask once there is a cut-off; these are the cut-offs.
ABSTENTION_THRESHOLDS: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)


@dataclass
class RecallResult:
    strategy: str
    per_question: dict[str, dict[int, float]] = field(default_factory=dict)
    hit_at_k: dict[int, list[float]] = field(default_factory=dict)
    recall_at_k: dict[int, list[float]] = field(default_factory=dict)
    misses: dict[int, list[str]] = field(default_factory=dict)
    # top-1 cosine distance per question, split by whether the corpus can answer it.
    top_distance_empty: dict[str, float] = field(default_factory=dict)
    top_distance_answerable: dict[str, float] = field(default_factory=dict)

    def abstention_curve(self) -> dict[float, dict[str, float]]:
        """For each threshold: how often it abstains correctly vs. how often it costs an answer.

        `correct_abstention` — expected-empty questions where even the nearest chunk is
        further than the threshold, so the system says "not in the corpus".
        `false_abstention`   — answerable questions the same threshold would silence.
        A threshold is only useful where the first is high and the second is near zero.
        """
        curve: dict[float, dict[str, float]] = {}
        for threshold in ABSTENTION_THRESHOLDS:
            empty = list(self.top_distance_empty.values())
            answerable = list(self.top_distance_answerable.values())
            curve[threshold] = {
                "correct_abstention": round(
                    statistics.fmean([1.0 if d > threshold else 0.0 for d in empty]), 4
                ) if empty else 0.0,
                "false_abstention": round(
                    statistics.fmean([1.0 if d > threshold else 0.0 for d in answerable]), 4
                ) if answerable else 0.0,
            }
        return curve

    def summary(self, ks: Sequence[int]) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "questions_scored": len(self.per_question),
            "answerable_questions": len(self.top_distance_answerable),
            "expected_empty_questions": len(self.top_distance_empty),
            "recall_at_k": {k: round(statistics.fmean(self.recall_at_k[k]), 4) for k in ks if self.recall_at_k.get(k)},
            "hit_at_k": {k: round(statistics.fmean(self.hit_at_k[k]), 4) for k in ks if self.hit_at_k.get(k)},
            "abstention_curve": {str(t): v for t, v in self.abstention_curve().items()},
            "total_misses_at_max_k": len(self.misses.get(max(ks), [])),
        }


def evaluate_recall(
    questions: Iterable[GoldenQuestion],
    *,
    ks: Sequence[int] = (1, 3, 5, 10, 20),
    strategy: str = "line",
    window_size: int = 1,
    max_distance: float | None = None,
) -> RecallResult:
    """recall@k over the golden set, for one chunking configuration.

    Recall is computed at *line* granularity: a window chunk that covers an expected
    line counts as retrieving that line, which is the only way the two strategies can
    be compared on the same axis.
    """
    label = strategy if strategy == "line" else f"{strategy}{window_size}"
    result = RecallResult(strategy=label)
    ordered_ks = sorted(ks)
    top_k = max(ordered_ks)

    for question in questions:
        hits = search(
            question.question,
            k=top_k,
            strategy=strategy,
            window_size=window_size,
            character=question.character,
            season=question.season,
            max_distance=max_distance,
        )
        retrieved_order = [h.line_id for h in hits]
        expected = set(question.expected_line_ids)
        result.per_question[question.id] = {}

        top_distance = hits[0].distance if hits else 1.0
        if question.is_empty_case:
            # Nothing *should* come back. Recall is undefined; what matters is how far
            # away the nearest chunk was, which is what a threshold could act on.
            result.top_distance_empty[question.id] = top_distance
            continue
        result.top_distance_answerable[question.id] = top_distance

        for k in ordered_ks:
            top = set(retrieved_order[:k])
            found = expected & top
            recall = len(found) / len(expected)
            result.recall_at_k.setdefault(k, []).append(recall)
            result.hit_at_k.setdefault(k, []).append(1.0 if found else 0.0)
            result.per_question[question.id][k] = recall
            if not found:
                result.misses.setdefault(k, []).append(question.id)

    return result


# ---------------------------------------------------------------------- faithfulness


@dataclass
class FaithfulnessResult:
    scored: int = 0
    grounded_citations: list[float] = field(default_factory=list)
    self_verified: list[float] = field(default_factory=list)
    answered_when_empty: list[float] = field(default_factory=list)
    per_question: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        def mean(values: list[float]) -> float | None:
            return round(statistics.fmean(values), 4) if values else None

        return {
            "questions_scored": self.scored,
            "citation_faithfulness": mean(self.grounded_citations),
            "self_verification_pass_rate": mean(self.self_verified),
            "correct_abstention_on_empty": mean(self.answered_when_empty),
        }


def cited_line_ids(text: str) -> list[int]:
    """Line ids the answer text itself claims to rely on."""
    return sorted({int(m) for m in _LINE_ID.findall(text)})


def evaluate_faithfulness(
    questions: Iterable[GoldenQuestion],
    *,
    model: str | None = None,
) -> FaithfulnessResult:
    """End-to-end: run the agent, then check what it cited against what it retrieved.

    Faithfulness here is deliberately mechanical — did every line the answer leans on
    actually come back from a retrieval tool in that same run? A fluent answer that
    cites a line the retriever never returned is the failure this catches.
    """
    from .agent import ask

    result = FaithfulnessResult()
    for question in questions:
        answer = ask(question.question, model=model)
        retrieved = set(answer.retrieved_line_ids)

        declared: set[int] = set()
        for call in answer.tool_calls:
            if call["name"] == "verify_answer":
                declared.update(int(i) for i in call["args"].get("cited_line_ids", []))
        in_text = set(cited_line_ids(answer.text))
        cited = declared | in_text

        grounded = len(cited & retrieved) / len(cited) if cited else 1.0
        result.grounded_citations.append(grounded)
        if answer.verified is not None:
            result.self_verified.append(1.0 if answer.verified else 0.0)
        if question.is_empty_case:
            # The right behaviour is to say the corpus cannot answer, citing nothing.
            result.answered_when_empty.append(1.0 if not cited else 0.0)

        result.scored += 1
        result.per_question.append(
            {
                "id": question.id,
                "question": question.question,
                "cited": sorted(cited),
                "retrieved": len(retrieved),
                "ungrounded": sorted(cited - retrieved),
                "citation_faithfulness": round(grounded, 4),
                "self_verified": answer.verified,
                "tools_used": answer.tools_used,
                "answer": answer.text,
            }
        )
    return result


# ------------------------------------------------------------------------- reporting


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(str(h) for h in headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def run_eval(
    *,
    golden_path: Path | None = None,
    ks: Sequence[int] = (1, 3, 5, 10, 20),
    strategy: str = "line",
    window_size: int = 1,
    with_agent: bool = False,
    limit: int | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    questions = load_golden(golden_path)
    if limit:
        questions = questions[:limit]

    recall = evaluate_recall(questions, ks=ks, strategy=strategy, window_size=window_size)
    summary = recall.summary(ks)

    print(f"golden set: {len(questions)} questions "
          f"({sum(1 for q in questions if q.is_empty_case)} expected-empty)")
    print(f"strategy:   {summary['strategy']}\n")
    print(_table(
        ["k", "recall@k", "hit@k", "misses"],
        [[k,
          summary["recall_at_k"].get(k, "-"),
          summary["hit_at_k"].get(k, "-"),
          len(recall.misses.get(k, []))] for k in sorted(ks)],
    ))
    curve = recall.abstention_curve()
    if recall.top_distance_empty:
        print("\nabstention: a distance cut-off is what lets the system say "
              "'not in this corpus'.")
        print(_table(
            ["max_distance", "correct abstention (n=%d)" % len(recall.top_distance_empty),
             "false abstention (n=%d)" % len(recall.top_distance_answerable)],
            [[t, v["correct_abstention"], v["false_abstention"]] for t, v in curve.items()],
        ))

    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "recall": summary,
        "misses": {str(k): v for k, v in recall.misses.items()},
        "top_distance_empty": recall.top_distance_empty,
        "top_distance_answerable": recall.top_distance_answerable,
    }

    if with_agent:
        print("\nrunning the agent end to end for faithfulness ...")
        faith = evaluate_faithfulness(questions)
        payload["faithfulness"] = faith.summary()
        payload["faithfulness_detail"] = faith.per_question
        print(_table(
            ["metric", "value"],
            [[k, v] for k, v in faith.summary().items()],
        ))

    target = output or (EVALS_DIR / "results.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nwrote {target}")
    return payload


def run_sweep(
    *,
    golden_path: Path | None = None,
    ks: Sequence[int] = (1, 3, 5, 10, 20),
    strategies: Sequence[str] = ("line", "window3", "window5"),
    output: Path | None = None,
) -> dict[str, Any]:
    """Sweep chunking strategy x top-k and tabulate. Builds missing chunk tables."""
    from .ingest.load import build_chunks, loaded_strategies

    questions = load_golden(golden_path)
    present = {(s, w) for s, w, _ in loaded_strategies()}

    results: list[RecallResult] = []
    for label in strategies:
        strategy, window = parse_strategy(label)
        if (strategy, window) not in present:
            print(f"building chunks for {label} ...")
            build_chunks(strategy=strategy, window_size=window, show_progress=True)
        results.append(
            evaluate_recall(questions, ks=ks, strategy=strategy, window_size=window)
        )

    ordered_ks = sorted(ks)
    rows = []
    for result in results:
        summary = result.summary(ordered_ks)
        rows.append([summary["strategy"]] + [summary["recall_at_k"].get(k, "-") for k in ordered_ks])

    print(f"\ngolden set: {len(questions)} questions\n")
    print("recall@k")
    print(_table(["strategy"] + [f"k={k}" for k in ordered_ks], rows))

    hit_rows = []
    for result in results:
        summary = result.summary(ordered_ks)
        hit_rows.append([summary["strategy"]] + [summary["hit_at_k"].get(k, "-") for k in ordered_ks])
    print("\nhit@k (at least one expected line retrieved)")
    print(_table(["strategy"] + [f"k={k}" for k in ordered_ks], hit_rows))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ks": list(ordered_ks),
        "strategies": list(strategies),
        "results": [r.summary(ordered_ks) for r in results],
        "misses": {r.strategy: {str(k): v for k, v in r.misses.items()} for r in results},
        "per_question": {r.strategy: {q: {str(k): v for k, v in d.items()}
                                      for q, d in r.per_question.items()} for r in results},
    }
    target = output or (EVALS_DIR / "sweep.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nwrote {target}")
    return payload
