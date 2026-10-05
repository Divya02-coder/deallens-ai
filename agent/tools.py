from document_engine.search import DocIndex


class Toolbox:
    """
    Collection of controlled tools available to the
    DealLens AI investigation agent.
    """

    def __init__(self, doc_index=None, state=None):
        self.docs = doc_index
        self.state = state or {}

    # =========================================================
    # DOCUMENT SEARCH
    # =========================================================

    def search_documents(
        self,
        query_text: str,
        k: int = 5,
    ):
        if not self.docs:
            return {
                "error": (
                    "No documents indexed. "
                    "Upload a PDF first."
                )
            }

        results = self.docs.search(
            query_text,
            k,
        )

        if not results:
            return {
                "query": query_text,
                "results": [],
                "message": (
                    "No relevant evidence found."
                ),
            }

        return {
            "query": query_text,
            "results": results,
        }

    # =========================================================
    # FINANCIAL ANALYSIS
    # =========================================================

    def get_financial_summary(self):
        """
        Return the financial analysis already calculated
        by the deterministic DealLens pipeline.
        """

        if not self.state:
            return {
                "error": "No analysis state is available."
            }

        result = {}

        # Keep the tool safe even if some keys don't exist.
        for key in [
            "financials",
            "ratios",
            "findings",
            "risks",
        ]:
            if key in self.state:
                result[key] = self.state[key]

        if not result:
            return {
                "message": (
                    "No financial summary was available "
                    "from the current analysis."
                )
            }

        return result

    # =========================================================
    # FINDINGS
    # =========================================================

    def get_findings(self):
        """
        Return deterministic DealLens findings.
        """

        findings = self.state.get(
            "findings",
            [],
        )

        return {
            "findings": findings
        }

    # =========================================================
    # GENERIC TOOL DISPATCHER
    # =========================================================

    def run(
        self,
        tool_name: str,
        arguments: dict,
    ):
        """
        Execute a tool requested by the AI agent.
        """

        if arguments is None:
            arguments = {}

        if tool_name == "search_documents":

            return self.search_documents(
                query_text=arguments.get(
                    "query_text",
                    arguments.get(
                        "query",
                        "",
                    ),
                ),
                k=int(
                    arguments.get(
                        "k",
                        5,
                    )
                ),
            )

        if tool_name == "get_financial_summary":

            return self.get_financial_summary()

        if tool_name == "get_findings":

            return self.get_findings()

        return {
            "error": (
                f"Unknown DealLens tool: "
                f"{tool_name}"
            )
        }


# =============================================================
# GEMINI TOOL SCHEMAS
# =============================================================

TOOL_SCHEMAS = [

    {
        "name": "search_documents",
        "description": (
            "Search indexed company documents using "
            "semantic similarity. Use this when the "
            "investigation needs evidence from uploaded "
            "PDFs or business documents."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query_text": {
                    "type": "STRING",
                    "description": (
                        "The question or topic to search "
                        "for in the indexed documents."
                    ),
                },
                "k": {
                    "type": "INTEGER",
                    "description": (
                        "Number of relevant document "
                        "chunks to retrieve."
                    ),
                },
            },
            "required": [
                "query_text"
            ],
        },
    },

    {
        "name": "get_financial_summary",
        "description": (
            "Retrieve the deterministic financial "
            "analysis already calculated by DealLens."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        },
    },

    {
        "name": "get_findings",
        "description": (
            "Retrieve risk findings generated by "
            "the deterministic DealLens analysis."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        },
    },
]