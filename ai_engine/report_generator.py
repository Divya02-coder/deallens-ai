from ai_engine.llm import GeminiLLM


class AIReportGenerator:

    def __init__(self):
        self.llm = GeminiLLM()

    # ---------------------------------------------------------
    # DETERMINISTIC FALLBACK
    # ---------------------------------------------------------

    def _fallback_report(self, state):
        findings = state.get("findings", [])

        report = []

        report.append("# DealLens AI — Due Diligence Report\n")

        report.append(
            "## Executive Summary\n"
        )

        report.append(
            "DealLens AI identified the following areas "
            "requiring analyst investigation. The findings "
            "are based on deterministic financial analysis, "
            "dependency analysis, anomaly screening and "
            "available document evidence.\n"
        )

        if not findings:

            report.append(
                "No significant findings were generated "
                "by the current analysis pipeline."
            )

            return "\n".join(report)

        report.append("## Key Findings\n")

        for i, finding in enumerate(findings, 1):

            severity = finding.get(
                "severity",
                "Unknown",
            )

            title = finding.get(
                "title",
                f"Finding {i}",
            )

            description = finding.get(
                "description",
                finding.get(
                    "message",
                    "No description available.",
                ),
            )

            report.append(
                f"### {i}. {title}\n"
            )

            report.append(
                f"**Severity:** {severity}\n\n"
            )

            report.append(
                f"{description}\n"
            )

            evidence = finding.get(
                "evidence"
            )

            if evidence:
                report.append(
                    f"\n**Evidence:** {evidence}\n"
                )

            report.append("")

        report.append(
            "## Analyst Follow-up\n"
        )

        report.append(
            "- Validate each high-severity finding "
            "against primary source documents.\n"
            "- Investigate unusual transactions before "
            "drawing conclusions.\n"
            "- Review major customer and supplier "
            "dependencies.\n"
            "- Review contract expiry and renewal exposure.\n"
            "- Validate financial assumptions with the "
            "target company's management and advisers."
        )

        return "\n".join(report)

    # ---------------------------------------------------------
    # AI REPORT
    # ---------------------------------------------------------

    def generate(
        self,
        state,
        additional_context=None,
    ):

        findings = state.get(
            "findings",
            [],
        )

        prompt = f"""
You are DealLens AI, an M&A financial due-diligence
investigation assistant.

Your job is to transform structured analysis into a concise,
professional analyst report.

IMPORTANT RULES:

1. Do not claim that an anomaly proves fraud.
2. Use language such as "potential risk", "anomaly",
   "requires investigation", or "exposure".
3. Do not invent financial values.
4. Distinguish calculated facts from interpretation.
5. Mention evidence whenever available.
6. Highlight the most material risks first.
7. Give practical questions an M&A analyst should investigate.
8. Keep the report professional and concise.

STRUCTURED FINDINGS:

{findings}

ADDITIONAL CONTEXT:

{additional_context or "No additional context provided."}

Create the report using this structure:

# DealLens AI — Due Diligence Report

## Executive Summary

## Key Financial Risks

## Customer & Supplier Dependencies

## Contract / Operational Exposure

## Transaction / Anomaly Signals

## Questions for Management

## Recommended Analyst Follow-up

End with a short disclaimer that the findings are
decision-support signals and require human validation.
"""

        try:

            return self.llm.generate(
                prompt=prompt,
                system_instruction=(
                    "You are an evidence-first M&A "
                    "financial analyst. Never invent evidence."
                ),
                max_tokens=3500,
                temperature=0.2,
            )

        except Exception as exc:

            fallback = self._fallback_report(state)

            return (
                fallback
                + "\n\n---\n"
                + "**AI interpretation temporarily unavailable.**\n"
                + f"Reason: {exc}\n"
                + "\nThe deterministic DealLens analysis "
                  "remains valid and available."
            )