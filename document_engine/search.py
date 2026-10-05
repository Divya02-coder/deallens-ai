"""
Semantic document search for DealLens AI.

Uses:
    Sentence Transformers -> embeddings
    ChromaDB              -> persistent vector store

The public interface intentionally remains:
    DocIndex(chunks)
    DocIndex.search(query, k)

so the rest of DealLens does not need to know
how retrieval is implemented.
"""

from pathlib import Path
import os

import chromadb
from sentence_transformers import SentenceTransformer


class DocIndex:

    def __init__(
        self,
        chunks: list[dict],
        persist_path: str | None = None,
        collection_name: str = "deallens_documents",
    ):

        self.chunks = chunks

        self.persist_path = persist_path or os.getenv(
            "VECTOR_DB_PATH",
            "data/vector_db",
        )

        Path(self.persist_path).mkdir(
            parents=True,
            exist_ok=True,
        )

        # Local embedding model.
        # No API key required.
        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.client = chromadb.PersistentClient(
            path=self.persist_path
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=collection_name
            )
        )

        self._index_chunks()

    # ---------------------------------------------------------
    # INDEX DOCUMENTS
    # ---------------------------------------------------------

    def _index_chunks(self):

        if not self.chunks:
            return

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(self.chunks):

            text = str(
                chunk.get("text", "")
            ).strip()

            if not text:
                continue

            source = str(
                chunk.get("source", "unknown")
            )

            page = int(
                chunk.get("page", 0)
            )

            # Stable ID prevents duplicates when
            # Streamlit reruns.
            chunk_id = (
                f"{source}__page_{page}__chunk_{i}"
            )

            ids.append(chunk_id)
            documents.append(text)

            metadatas.append({
                "source": source,
                "page": page,
            })

        if not documents:
            return

        embeddings = self.embedding_model.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    # ---------------------------------------------------------
    # SEMANTIC SEARCH
    # ---------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> list[dict]:

        if not query.strip():
            return []

        if self.collection.count() == 0:
            return []

        query_embedding = (
            self.embedding_model.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
            .tolist()
        )

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        documents = results.get(
            "documents",
            [[]],
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]],
        )[0]

        distances = results.get(
            "distances",
            [[]],
        )[0]

        output = []

        for text, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):

            # Chroma returns distance.
            # Lower distance = better match.
            similarity = max(
                0.0,
                min(
                    1.0,
                    1.0 - float(distance),
                ),
            )

            output.append({
                "source": metadata.get(
                    "source",
                    "unknown",
                ),
                "page": metadata.get(
                    "page",
                    0,
                ),
                "text": text,
                "score": similarity,
            })

        return output