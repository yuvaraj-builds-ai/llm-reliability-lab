# LLM Reliability Lab

An evaluation-first RAG assistant that tests retrieval, citations, refusals, prompt injection, and response latency.

> NovaCart is a fictional company. All policies and evaluation data were created solely for this demonstration.

## Business problem

Fluent answers can still be wrong. This project checks whether a policy assistant retrieves evidence, cites the correct documents, corrects false assumptions, and refuses unsupported requests.

## What the project demonstrates

- Local semantic retrieval over ten fictional policies
- Evidence-only prompting and structured response validation
- Exact-filename citation validation
- Fifty evaluation cases across five reliability categories
- Clear automated metrics plus honest manual answer review

## Architecture

```mermaid
flowchart LR
    Q[User Question] --> E[Question Embedding]
    E --> F[FAISS Retrieval]
    F --> C[Top 4 Policy Chunks]
    C --> L[LLM]
    L --> V[Schema and Citation Validation]
    V --> A[Answer with Sources]
```

## Features

The app displays the answer, citations, latency, and retrieved evidence. Unsupported questions use a fixed refusal. Model output is parsed with Pydantic and retried once if malformed.

## Technology stack

Python 3.11+, Streamlit, Sentence Transformers (`all-MiniLM-L6-v2`), FAISS CPU, OpenAI Python SDK, Pydantic, Pandas, pytest, and python-dotenv.

Python 3.11 or newer is required. The current release was tested with Python 3.13.

## Repository structure

`src/` contains focused application modules; `knowledge_base/` contains policies; `data/` contains cases and the generated index; `scripts/` contains entry points; `tests/` contains offline tests; and `reports/` receives evaluation output.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
python -m pip install -r requirements.txt
copy .env.example .env
```

Fill all three `.env` values for one OpenAI-compatible endpoint. Never commit `.env`.

## Build the index

```bash
python scripts/build_index.py
```

The first run downloads the local embedding model. Later app starts reuse the persisted index.

## Run the application

```bash
streamlit run streamlit_app.py
```

## Run the evaluation

```bash
python scripts/run_evaluation.py
```

This creates `reports/evaluation_results.csv` and `reports/evaluation_summary.json`.

## Run the tests

```bash
pytest -q
```

Tests make no LLM API calls.

## Evaluation methodology

Retrieval Hit@4 passes when at least one expected source is retrieved. Refusal accuracy compares expected and actual refusal. Citation accuracy requires citations to be retrieved and include an expected source for supported cases. Output validity records schema success. Total and p95 latency are recorded.

Answer correctness remains a manual review: `2` means correct and complete, `1` partially correct, and `0` incorrect or misleading. The CSV leaves `manual_correctness_score` and `manual_review_notes` blank. No LLM judge is used.

## Results

Run the evaluation against your configured endpoint to produce actual results. This repository does not claim unmeasured accuracy.

The automated evaluation achieved 100% retrieval, refusal, citation-validation, and structured-output success on a predefined 50-case fictional benchmark during the latest local run. These figures measure this fixed benchmark and pipeline configuration; they do not represent general chatbot accuracy. Semantic answer quality must be evaluated separately through the documented manual review.

| Metric | Result |
| --- | ---: |
| Retrieval Hit@4 | 100% |
| Refusal Accuracy | 100% |
| Citation Validation | 100% |
| Structured Output Success | 100% |
| Manual Answer Quality | Pending review |
| Average Latency | 1.51 seconds |
| p95 Latency | 2.51 seconds |
| Offline Tests | 12 passing |
## Example failures

Useful failures include retrieval selecting a related but wrong policy, a supported answer omitting a source, or an unsupported prompt producing a plausible invention. Inspect retrieved chunks and manually score the answer before changing prompts.

## Limitations

The small fictional corpus does not represent production scale. Dense retrieval has no reranker or keyword fallback. Endpoint behavior varies, citation validation checks source provenance rather than factual entailment, and manual review is still required.

## Privacy and data

The repository contains only fictional policies and evaluation questions. Questions sent through the app are transmitted to the configured LLM endpoint; consult that endpoint's privacy terms.

## Future improvements

Only after establishing baseline results: analyze failures, improve weak cases or policy wording, then rerun the same evaluation to measure the change.

## License

MIT. See [LICENSE](LICENSE).
