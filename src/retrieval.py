import json
from typing import Any
from src.config import INDEX_DIR, INDEX_PATH, METADATA_PATH, MODEL_NAME, TOP_K
from src.knowledge import load_knowledge_base
from src.schemas import Chunk, RetrievedChunk

_model: Any = None


def get_embedding_model() -> Any:
    """Load the local embedding model once per Python process."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def set_embedding_model(model: Any) -> None:
    """Provide a cached model instance, primarily for Streamlit reruns and tests."""
    global _model
    _model = model

def build_index() -> int:
    import faiss
    chunks = load_knowledge_base()
    vectors = get_embedding_model().encode([c.text for c in chunks], normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1]); index.add(vectors)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    METADATA_PATH.write_text(json.dumps([c.model_dump() for c in chunks], indent=2), encoding="utf-8")
    return len(chunks)

def retrieve(question: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
    if not question.strip(): raise ValueError("Question cannot be empty.")
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Index missing. Run: python scripts/build_index.py")
    import faiss
    index = faiss.read_index(str(INDEX_PATH))
    chunks = [Chunk.model_validate(x) for x in json.loads(METADATA_PATH.read_text(encoding="utf-8"))]
    query = get_embedding_model().encode([question], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(query, min(top_k, index.ntotal))
    return [RetrievedChunk(**chunks[i].model_dump(), score=float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]
