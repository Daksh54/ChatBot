# Showcase Domain Test

Use this checklist to validate `NexusRAG` against a demanding quantitative-research corpus once you have the source files.

## Recommended Corpus

- 500-stock universe reports
- Intraday execution analysis PDFs
- Strategy notes with formulas and factor definitions
- CSV/XLSX files containing returns, slippage, turnover, and signal diagnostics

## Validation Flow

1. Create or open the `Flagship Workspace`.
2. Upload a mix of large PDF reports and structured market data files.
3. Wait for each ingestion task to reach `SUCCEEDED`.
4. Run prompts that force cross-document retrieval, formula lookup, and temporal synthesis.

## Stress-Test Prompts

- "Compare the execution assumptions used across the uploaded intraday strategy reports."
- "Extract any formulas used for ranking, volatility targeting, or position sizing."
- "Which report discusses turnover constraints and how do those constraints affect the 500-stock universe?"
- "Summarize the slippage and execution-cost logic across the uploaded data and cite the exact sources."
- "Cross-reference the structured CSV metrics against the narrative PDF commentary and explain where the conclusions agree or diverge."

## Success Criteria

- Citations land on the correct document and page
- The assistant keeps multi-turn continuity across follow-up questions
- Long-running uploads do not block the API request cycle
- Hybrid retrieval surfaces the right formula-bearing chunks even when the query uses different wording
- Structured numeric summaries remain consistent with the uploaded CSV/XLSX content
