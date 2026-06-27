import time
from src.llm import generate_answer
from src.retrieval import retrieve
from src.schemas import AnswerResponse, RetrievedChunk

def format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[Source: {c.source} | Heading: {c.heading}]\n{c.text}" for c in chunks)

def validate_citations(answer: AnswerResponse, chunks: list[RetrievedChunk]) -> bool:
    return set(answer.citations) <= {c.source for c in chunks}

def answer_question(question: str) -> tuple[AnswerResponse, list[RetrievedChunk], float]:
    if not question.strip(): raise ValueError("Please enter a question.")
    start = time.perf_counter()
    chunks = retrieve(question)
    answer = generate_answer(question, format_context(chunks))
    if not validate_citations(answer, chunks):
        raise ValueError("The model returned a citation outside the retrieved context.")
    return answer, chunks, time.perf_counter() - start
