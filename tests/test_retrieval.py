import json, sys
from types import SimpleNamespace
import numpy as np
from src.schemas import Chunk, RetrievedChunk
from src import retrieval
def test_retrieved_chunk_metadata():
    chunk=RetrievedChunk(id="1",text="x",source="returns.md",heading="Rules",score=.9)
    assert chunk.source=="returns.md"

def test_retrieve_top_k_and_source(monkeypatch, tmp_path):
    index_path, metadata_path = tmp_path/"x.faiss", tmp_path/"chunks.json"
    index_path.touch()
    chunks=[Chunk(id=str(i),text="text",source=("returns.md" if i==0 else "shipping.md"),heading="Rules") for i in range(4)]
    metadata_path.write_text(json.dumps([c.model_dump() for c in chunks]),encoding="utf-8")
    class Index:
        ntotal=4
        def search(self, query, count):
            return np.array([[.9,.8,.7,.6]],dtype="float32")[:,:count], np.array([[0,1,2,3]])[:,:count]
    monkeypatch.setattr(retrieval,"INDEX_PATH",index_path)
    monkeypatch.setattr(retrieval,"METADATA_PATH",metadata_path)
    monkeypatch.setattr(retrieval,"get_embedding_model",lambda: SimpleNamespace(encode=lambda *a,**k:np.array([[1,0]],dtype="float32")))
    monkeypatch.setitem(sys.modules,"faiss",SimpleNamespace(read_index=lambda _:Index()))
    results=retrieval.retrieve("opened laptop",top_k=2)
    assert len(results)==2 and results[0].source=="returns.md"
