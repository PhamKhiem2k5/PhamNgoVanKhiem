import glob
import json
import os
import faiss
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PDF_DIR = "data/pdfs"
INDEX_PATH = "data/faiss_index.bin"
METADATA_PATH = "data/chunks_metadata.json"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def chunk_text(text, chunk_size=250, overlap=50):
  """Chia văn bản thành các đoạn nhỏ theo số từ (có gối đầu overlap)."""
  words = text.split()
  if not words:
    return []
  chunks = []
  step = chunk_size - overlap
  for i in range(0, len(words), step):
    chunk = " ".join(words[i : i + chunk_size])
    if len(chunk.strip()) > 30:  # Bỏ qua các đoạn quá ngắn/rác
      chunks.append(chunk)
  return chunks


def build_index():
  pdf_files = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
  if not pdf_files:
    raise FileNotFoundError(
        f"Không tìm thấy file PDF nào trong {PDF_DIR}. Hãy chạy"
        " download_data.py trước!"
    )

  all_chunks = []
  print(f"Đang trích xuất văn bản từ {len(pdf_files)} file PDF...")

  for pdf_path in tqdm(pdf_files):
    filename = os.path.basename(pdf_path)
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
      page_text = doc[page_num].get_text("text")
      # Làm sạch khoảng trắng thừa và dấu gạch nối ngắt dòng
      clean_text = " ".join(page_text.replace("-\n", "").split())
      page_chunks = chunk_text(clean_text, chunk_size=250, overlap=50)

      for c_text in page_chunks:
        all_chunks.append({
            "chunk_id": len(all_chunks),
            "source": filename,
            "page": page_num + 1,
            "text": c_text,
        })
    doc.close()

  print(
      f"Tổng số đoạn văn (chunks) tạo được: {len(all_chunks)}. Đang tạo"
      " embeddings..."
  )
  embedder = SentenceTransformer(EMBEDDING_MODEL)

  texts = [item["text"] for item in all_chunks]
  # normalize_embeddings=True giúp Inner Product (IndexFlatIP) tương đương Cosine Similarity
  embeddings = embedder.encode(
      texts,
      batch_size=32,
      show_progress_bar=True,
      normalize_embeddings=True,
  )
  embeddings = np.array(embeddings, dtype="float32")

  # Khởi tạo FAISS Index và lưu xuống ổ cứng
  dimension = embeddings.shape[1]
  index = faiss.IndexFlatIP(dimension)
  index.add(embeddings)

  faiss.write_index(index, INDEX_PATH)
  with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

  print(f"\nHoàn tất! Đã lưu FAISS index tại '{INDEX_PATH}' ({index.ntotal} vectors).")


if __name__ == "__main__":
  build_index()