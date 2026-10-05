"""
RAG retrieval layer.

Retrieves relevant evidence from the local
ChromaDB vector store.
"""


class DealLensRetriever:

    def __init__(
        self,
        doc_index=None,
    ):

        self.doc_index = doc_index

    def retrieve(
        self,
        question: str,
        k: int = 5,
    ):

        if not self.doc_index:

            return []

        return self.doc_index.search(
            question,
            k=k,
        )

    def build_context(
        self,
        question: str,
        k: int = 5,
    ):

        results = self.retrieve(
            question,
            k,
        )

        if not results:

            return (
                "No relevant document evidence "
                "was found.",
                [],
            )

        context = []

        for i, result in enumerate(
            results,
            start=1,
        ):

            context.append(
                f"""
SOURCE {i}
File: {result['source']}
Page: {result['page']}
Relevance: {result['score']:.3f}

{result['text']}
"""
            )

        return (
            "\n".join(context),
            results,
        )