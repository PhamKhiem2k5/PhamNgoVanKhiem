import json
from src.generator import RAGGenerator
from src.retriever import RAGRetriever


class RAGPipeline:

  def __init__(self):
    self.retriever = RAGRetriever()
    self.generator = RAGGenerator()

  def ask(
      self,
      question: str,
      top_k: int = 3,
      use_reranker: bool = True,
      lang: str = "en",
  ):
    chunks = self.retriever.retrieve(
        query=question, top_k=top_k, use_reranker=use_reranker
    )
    answer, sources = self.generator.generate(
        query=question, retrieved_chunks=chunks, lang=lang
    )
    return {"answer": answer, "sources": sources, "chunks": chunks}