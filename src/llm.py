import json
import re
from pydantic import ValidationError
from src.config import REFUSAL, llm_config
from src.schemas import AnswerResponse

SYSTEM = f"""You are NovaCart's policy assistant. Use only supplied context. Never invent policies or use outside knowledge. Correct false assumptions. If evidence is insufficient answer exactly: {REFUSAL} Cite exact source filenames for every factual claim. Return only JSON with answer, citations, refused. Refusals have no citations."""


def context_sources(context: str) -> set[str]:
    """Extract the exact filenames made available in the retrieved context."""
    return set(re.findall(r"\[Source:\s*([^|\]]+?)\s*\|", context))


def parse_answer(raw: str) -> AnswerResponse:
    """Parse model JSON, repairing only an omitted deterministic refusal flag."""
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("The model response must be a JSON object.")
    if "answer" in payload and "citations" in payload:
        is_exact_refusal = (
            payload["answer"].strip() == REFUSAL and payload["citations"] == []
        )
        if is_exact_refusal:
            # The fixed refusal text and empty citations make this boolean
            # deterministic, even when a compatible endpoint returns it wrong.
            payload["refused"] = True
        elif "refused" not in payload:
            payload["refused"] = False
    return AnswerResponse.model_validate(payload)


def generate_answer(question: str, context: str) -> AnswerResponse:
    from openai import OpenAI

    key, url, model = llm_config()
    client = OpenAI(api_key=key, base_url=url)
    allowed_sources = context_sources(context)
    allowed_text = ", ".join(sorted(allowed_sources)) or "none"
    messages = [
        {"role":"system","content":SYSTEM},
        {
            "role":"user",
            "content":(
                f"ALLOWED CITATION FILENAMES: {allowed_text}\n"
                "The citations array may contain only filenames from this list.\n\n"
                f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
            ),
        },
    ]
    error = None
    for _ in range(2):
        try:
            result = client.chat.completions.create(model=model, messages=messages, temperature=0, response_format={"type":"json_object"})
            raw = result.choices[0].message.content or ""
            answer = parse_answer(raw)
            invalid = set(answer.citations) - allowed_sources
            if invalid:
                raise ValueError(
                    f"Citations must use only these filenames: {allowed_text}"
                )
            return answer
        except (ValueError, json.JSONDecodeError, ValidationError) as exc:
            error = exc
            messages += [
                {"role":"assistant","content":locals().get("raw","")},
                {
                    "role":"user",
                    "content":(
                        "Correct the response. Return valid JSON only. "
                        'Required shape: {"answer":"text","citations":["source.md"],"refused":false}. '
                        f"The citations array may contain only: {allowed_text}. "
                        "If the context is insufficient, use the required refusal with no citations."
                    ),
                },
            ]
    raise ValueError(f"Invalid structured output after retry: {error}")
