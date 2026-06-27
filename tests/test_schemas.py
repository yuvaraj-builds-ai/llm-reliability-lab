import pytest
from pydantic import ValidationError
from src.schemas import AnswerResponse
from src.llm import context_sources, parse_answer
def test_valid(): assert AnswerResponse(answer="Yes",citations=["x.md"],refused=False)
def test_refusal_has_no_citations():
    with pytest.raises(ValidationError): AnswerResponse(answer="No",citations=["x.md"],refused=True)
def test_missing_field_rejected():
    with pytest.raises(ValidationError): AnswerResponse.model_validate({"answer":"x","citations":[]})
def test_context_sources_are_exact():
    context="[Source: returns.md | Heading: Rules]\nText\n\n[Source: shipping.md | Heading: Timing]\nText"
    assert context_sources(context)=={"returns.md","shipping.md"}
def test_missing_refused_flag_is_repaired_for_supported_answer():
    answer=parse_answer('{"answer":"Yes.","citations":["returns.md"]}')
    assert answer.refused is False
def test_incorrect_refused_flag_is_repaired_for_exact_refusal():
    from src.config import REFUSAL
    raw=f'{{"answer":{__import__("json").dumps(REFUSAL)},"citations":[],"refused":false}}'
    assert parse_answer(raw).refused is True
