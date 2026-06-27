import pandas as pd
from src.evaluation import retrieval_hit, refusal_correct, citation_correct, summarize
def test_metrics():
    assert retrieval_hit(["a.md"],["a.md","b.md"])
    assert refusal_correct(True,True)
    assert citation_correct(["a.md"],["a.md"],["a.md"],False)
    assert not citation_correct(["fake.md"],["a.md"],["a.md"],False)
def test_summary():
    frame=pd.DataFrame([{"category":"normal","retrieval_hit":True,"refusal_correct":True,"citation_correct":True,"output_valid":True,"latency_seconds":1.0}])
    result=summarize(frame)
    assert result["total_cases"]==1 and result["retrieval_hit_rate"]==1.0
