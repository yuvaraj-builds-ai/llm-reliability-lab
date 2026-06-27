import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = ROOT / "knowledge_base"
INDEX_DIR = ROOT / "data" / "index"
INDEX_PATH = INDEX_DIR / "novacart.faiss"
METADATA_PATH = INDEX_DIR / "chunks.json"
CASES_PATH = ROOT / "data" / "evaluation_cases.jsonl"
REPORTS_DIR = ROOT / "reports"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 4
REFUSAL = "I do not have enough information in the NovaCart knowledge base to answer that question."

def llm_config() -> tuple[str, str, str]:
    load_dotenv(ROOT / ".env")
    values = tuple(os.getenv(k, "").strip() for k in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"))
    if not all(values):
        raise ValueError("Missing LLM configuration. Copy .env.example to .env and fill in all three values.")
    return values
