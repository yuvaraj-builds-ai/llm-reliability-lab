import hashlib
import re
from pathlib import Path
from src.config import KNOWLEDGE_DIR
from src.schemas import Chunk

def discover_markdown_files(folder: Path = KNOWLEDGE_DIR) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Knowledge-base folder not found: {folder}")
    return sorted(p for p in folder.glob("*.md") if p.stat().st_size)

def chunk_document(path: Path, max_chars: int = 1800) -> list[Chunk]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    matches = list(re.finditer(r"^#{1,6}\s+(.+)$", text, re.M))
    sections = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end():end].strip()
        if body:
            sections.append((match.group(1).strip(), body))
    if not matches:
        sections = [("Document", text)]
    chunks = []
    for heading, body in sections:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        parts, current = [], ""
        for p in paragraphs:
            if current and len(current) + len(p) > max_chars:
                parts.append(current); current = ""
            current = f"{current}\n\n{p}".strip()
        if current: parts.append(current)
        for n, part in enumerate(parts):
            ident = hashlib.sha1(f"{path.name}{heading}{n}".encode()).hexdigest()[:12]
            chunks.append(Chunk(id=ident, text=f"{heading}\n\n{part}", source=path.name, heading=heading))
    return chunks

def load_knowledge_base(folder: Path = KNOWLEDGE_DIR) -> list[Chunk]:
    return [c for p in discover_markdown_files(folder) for c in chunk_document(p)]
