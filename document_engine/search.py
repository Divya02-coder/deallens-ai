"""Evidence index for DealLens.

Uses a lightweight TF-IDF index by default so the app stays easy to run locally.
The public interface is intentionally simple: DocIndex(chunks).search(query, k).
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class DocIndex:
    def __init__(
        self,
        chunks,
        persist_path=None,
        collection_name="documents",
        **kwargs,
    ):
        self.persist_path = persist_path
        self.collection_name = collection_name
        self.chunks = list(chunks or [])
        texts = [str(c.get("text", "")) for c in self.chunks]
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=20000)
        self.mat = self.vec.fit_transform(texts) if any(t.strip() for t in texts) else None

    def search(self, query: str, k: int = 5) -> list[dict]:
        query = str(query or "").strip()
        if not query or not self.chunks or self.mat is None:
            return []
        sims = cosine_similarity(self.vec.transform([query]), self.mat).ravel()
        idx = sims.argsort()[::-1][: max(1, int(k))]
        return [{**self.chunks[i], "score": float(sims[i])} for i in idx if sims[i] > 0]
