"""STEP 10 - Document search with citations.
TF-IDF baseline (no API key needed). Swap for embeddings + Chroma/Qdrant/pgvector later - same interface."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class DocIndex:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.mat = self.vec.fit_transform([c["text"] for c in chunks]) if chunks else None

    def search(self, query: str, k: int = 5) -> list[dict]:
        if not self.chunks:
            return []
        sims = cosine_similarity(self.vec.transform([query]), self.mat).ravel()
        idx = sims.argsort()[::-1][:k]
        return [{**self.chunks[i], "score": float(sims[i])} for i in idx if sims[i] > 0]
