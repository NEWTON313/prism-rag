"""Observabilité et boucle d'amélioration (leçon 5.4).

Chaque requête produit une TRACE : question, passages retrouvés et leurs scores, réponse, sources citées,
durées par étape, retour de l'utilisateur. Les traces alimentent trois analyses : les questions sans réponse
(trous du corpus), les réponses mal notées (à rejouer), et les nouvelles questions du jeu d'évaluation.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Trace:
    question: str
    retrieved: list[tuple[str, float]] = field(default_factory=list)
    answer: str = ""
    cited: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    feedback: int | None = None  # +1 / -1
    cache_hit: bool = False

    @contextmanager
    def step(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings_ms[name] = round((time.perf_counter() - start) * 1000, 1)

    @property
    def unanswered(self) -> bool:
        return not self.cited


class TraceLog:
    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, trace: Trace) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(trace), ensure_ascii=False) + "\n")

    def read(self) -> list[Trace]:
        if not self.path.exists():
            return []
        traces = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            raw = json.loads(line)
            raw["retrieved"] = [tuple(x) for x in raw["retrieved"]]
            traces.append(Trace(**raw))
        return traces


def weekly_review(traces: list[Trace]) -> dict:
    """Le rapport à relire chaque semaine."""
    answered = [t for t in traces if not t.unanswered]
    rated = [t for t in traces if t.feedback is not None]
    latencies = sorted(sum(t.timings_ms.values()) for t in traces) or [0.0]
    return {
        "requetes": len(traces),
        "sans_reponse": [t.question for t in traces if t.unanswered],
        "mal_notees": [t.question for t in rated if t.feedback < 0],
        "taux_satisfaction": sum(t.feedback > 0 for t in rated) / len(rated) if rated else None,
        "taux_cache": sum(t.cache_hit for t in traces) / len(traces) if traces else 0.0,
        "latence_p95_ms": latencies[int(0.95 * (len(latencies) - 1))],
        "articles_les_plus_cites": Counter(a for t in answered for a in t.cited).most_common(5),
    }
