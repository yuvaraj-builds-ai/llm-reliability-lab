from pathlib import Path
from src.knowledge import discover_markdown_files, chunk_document
def test_discovers_and_preserves_source(tmp_path: Path):
    (tmp_path/"x.md").write_text("# One\n\nText",encoding="utf-8")
    assert discover_markdown_files(tmp_path)[0].name=="x.md"
    assert chunk_document(tmp_path/"x.md")[0].source=="x.md"
def test_empty_ignored(tmp_path: Path):
    (tmp_path/"x.md").write_text("",encoding="utf-8")
    assert discover_markdown_files(tmp_path)==[]
