"""BM25 (leçon 3.1) : la recherche lexicale, écrite en 40 lignes pour en comprendre chaque terme.

    score(q, d) = Σ_t IDF(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 − b + b·|d|/avgdl))

- IDF récompense les mots rares (« E02 » pèse plus que « bureau ») ;
- k1 sature la répétition d'un mot ; b pénalise les documents longs.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

STOPWORDS = set("""au aux avec ce ces dans de des du elle en et eux il je la le les leur lui ma mais me même mes moi
mon ne nos notre nous on ou par pas pour qu que qui sa se ses son sur ta te tes toi ton tu un une vos votre vous
c d j l à m n s t y été est sont être avoir ai as a avons avez ont fait faut peut plus comment quel quelle quels
quelles combien est-ce si""".split())
TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")


def tokenize(text: str) -> list[str]:
    """Minuscules, sans accents, sans mots vides, racine grossière (6 lettres).

    « garanti(e)(s) » → « garant » ; un terme contenant des chiffres (« E02 ») est gardé tel quel.
    """
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    terms = [t for t in TOKEN.findall(text) if t not in STOPWORDS]
    return [t if any(ch.isdigit() for ch in t) else t[:6] for t in terms]


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avgdl = sum(self.lengths) / max(len(self.docs), 1)
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for doc, length in zip(self.docs, self.lengths):
            s = 0.0
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avgdl))
            out.append(s)
        return out
