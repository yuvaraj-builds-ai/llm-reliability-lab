import json
from pathlib import Path
import pandas as pd
from src.config import CASES_PATH, REPORTS_DIR
from src.rag import answer_question

def retrieval_hit(expected: list[str], retrieved: list[str]) -> bool:
    return not expected or bool(set(expected) & set(retrieved))
def refusal_correct(expected: bool, actual: bool) -> bool:
    return expected == actual
def citation_correct(citations: list[str], retrieved: list[str], expected: list[str], refused: bool) -> bool:
    return not citations if refused else set(citations) <= set(retrieved) and bool(set(citations) & set(expected))
def summarize(frame: pd.DataFrame) -> dict:
    result = {"total_cases":len(frame)}
    for col, name in [("retrieval_hit","retrieval_hit_rate"),("refusal_correct","refusal_accuracy"),("citation_correct","citation_accuracy"),("output_valid","structured_output_success_rate")]:
        result[name] = float(frame[col].mean())
    result["average_latency"] = float(frame.latency_seconds.mean())
    result["p95_latency"] = float(frame.latency_seconds.quantile(.95))
    result["by_category"] = frame.groupby("category")[["retrieval_hit","refusal_correct","citation_correct","output_valid"]].mean().to_dict("index")
    return result
def run_evaluation(path: Path = CASES_PATH) -> dict:
    cases = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    rows = []
    for case in cases:
        try:
            answer, chunks, latency = answer_question(case["question"])
            sources = [c.source for c in chunks]
            rows.append({**case,"retrieved_sources":";".join(sources),"answer":answer.answer,"citations":";".join(answer.citations),"refused":answer.refused,"retrieval_hit":retrieval_hit(case["expected_sources"],sources),"refusal_correct":refusal_correct(case["should_refuse"],answer.refused),"citation_correct":citation_correct(answer.citations,sources,case["expected_sources"],answer.refused),"output_valid":True,"latency_seconds":latency,"manual_correctness_score":"","manual_review_notes":""})
        except Exception as exc:
            rows.append({**case,"error":str(exc),"retrieval_hit":False,"refusal_correct":False,"citation_correct":False,"output_valid":False,"latency_seconds":0,"manual_correctness_score":"","manual_review_notes":""})
    frame = pd.DataFrame(rows); summary = summarize(frame)
    REPORTS_DIR.mkdir(exist_ok=True)
    frame.to_csv(REPORTS_DIR/"evaluation_results.csv", index=False)
    (REPORTS_DIR/"evaluation_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary
