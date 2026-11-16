import json
import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

INDEX_PATH = "data/faiss_index.bin"
METADATA_PATH = "data/chunks_metadata.json"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


class RAGRetriever:

  def __init__(self):
    self.index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
      self.chunks = json.load(f)

    self.embedder = SentenceTransformer(EMBEDDING_MODEL)
    self.reranker = CrossEncoder(RERANKER_MODEL)

  def retrieve(
      self,
      query: str,
      top_k: int = 3,
      use_reranker: bool = True,
      initial_k: int = 20,
  ):
    query_vec = self.embedder.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype="float32")

    # Nếu không dùng Reranker (Baseline), chỉ cần tìm trực tiếp top_k từ FAISS
    fetch_k = initial_k if use_reranker else top_k
    fetch_k = min(fetch_k, len(self.chunks))

    scores, indices = self.index.search(query_vec, fetch_k)

    candidates = []
    for score, idx in zip(scores[0], indices[0]):
      if idx == -1:
        continue
      item = dict(self.chunks[idx])
      item["retrieval_score"] = float(score)
      candidates.append(item)

    if not use_reranker or not candidates:
      return candidates[:top_k]

    # Bước Reranking: chấm điểm lại từng cặp (query, passage)
    pairs = [(query, c["text"]) for c in candidates]
    rerank_scores = self.reranker.predict(pairs)

    for c, r_score in zip(candidates, rerank_scores):
      c["rerank_score"] = float(r_score)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]