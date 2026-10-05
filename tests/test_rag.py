from document_engine.search import DocIndex


def test_semantic_document_search():

    chunks = [
        {
            "source": "contract_a.pdf",
            "page": 5,
            "text": (
                "The agreement contains a "
                "change of control clause."
            ),
        },
        {
            "source": "contract_b.pdf",
            "page": 12,
            "text": (
                "The supplier agreement "
                "expires in December."
            ),
        },
    ]

    index = DocIndex(
        chunks,
        persist_path="data/vector_db_test",
        collection_name="test_documents",
    )

    results = index.search(
        "Can the contract change when ownership changes?",
        k=1,
    )

    assert len(results) == 1

    assert (
        results[0]["source"]
        == "contract_a.pdf"
    )

    assert (
        results[0]["page"]
        == 5
    )