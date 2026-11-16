import json
import os
import time
import numpy as np
import pymupdf
import streamlit as st
from src.ingest import chunk_text
from src.pipeline import RAGPipeline

st.set_page_config(page_title="RAG Document Chat", page_icon="📄", layout="wide")


@st.cache_resource
def load_rag_pipeline():
  return RAGPipeline()


rag = load_rag_pipeline()

# Từ điển đa ngôn ngữ đồng bộ toàn bộ giao diện
TEXTS = {
    "vi": {
        "sidebar_config": "⚙️ Cấu hình Hệ thống",
        "use_reranker": "Sử dụng bộ xếp hạng lại (Reranker)",
        "top_k": "Số đoạn văn trích xuất (Top-k)",
        "upload_header": "📤 Thêm tài liệu PDF mới",
        "upload_label": "Chọn hoặc kéo thả file PDF vào khung bên dưới",
        "index_btn": "Lập chỉ mục tài liệu PDF",
        "indexing_spinner": "Đang đọc và tạo vector cho tài liệu PDF mới...",
        "indexing_success": (
            "Đã thêm {chunks} đoạn văn từ {files} tài liệu PDF!"
        ),
        "total_chunks": "Tổng số đoạn văn đang lưu trữ: {count} đoạn",
        "clear_btn": "🗑️ Xóa lịch sử trò chuyện",
        "title": "📄 Hỏi-Đáp Tài Liệu PDF với RAG & Reranking",
        "subtitle": (
            "Hệ thống hỏi-đáp trên bộ sưu tập PDF có trích dẫn nguồn chính xác"
            " (Embedding + Reranker + Open LLM)"
        ),
        "sample_expander": (
            "💡 Câu hỏi gợi ý từ bộ dữ liệu PDF hiện có (tiếng Anh)"
        ),
        "chat_placeholder": "Nhập câu hỏi về nội dung tài liệu PDF...",
        "mode_main": "Mô hình chính (Có Reranker)",
        "mode_base": "Mô hình cơ sở (Không Reranker)",
        "thinking": "Đang truy xuất tài liệu và tạo câu trả lời [{mode}]...",
        "time_caption": "⏱️ Thời gian xử lý: {elapsed:.2f} giây | Chế độ: {mode}",
        "sources_label": "📚 Nguồn trích dẫn ({count} đoạn văn)",
        "page_label": "Trang",
        "rerank_score": "Điểm xếp hạng lại",
        "cosine_score": "Điểm tương đồng",
        "file_label": "Tài liệu",
    },
    "en": {
        "sidebar_config": "⚙️ System Configuration",
        "use_reranker": "Use Cross-Encoder Reranker",
        "top_k": "Number of retrieved passages (Top-k)",
        "upload_header": "📤 Add New PDF Documents",
        "upload_label": "Select or drag and drop PDF files below",
        "index_btn": "Index Uploaded PDFs",
        "indexing_spinner": "Parsing and embedding new PDF files...",
        "indexing_success": (
            "Added {chunks} passages from {files} PDF file(s)!"
        ),
        "total_chunks": "Total indexed knowledge base: {count} passages",
        "clear_btn": "🗑️ Clear Chat History",
        "title": "📄 Document Chat with RAG & Reranking",
        "subtitle": (
            "Question Answering over PDF collections with verifiable citations"
            " (Embedding + Cross-Encoder Reranker + Open LLM)"
        ),
        "sample_expander": "💡 Sample questions from the current PDF dataset",
        "chat_placeholder": "Ask a question about your PDF documents...",
        "mode_main": "Main Model (With Reranker)",
        "mode_base": "Baseline (No Reranker)",
        "thinking": "Retrieving context and generating answer [{mode}]...",
        "time_caption": "⏱️ Latency: {elapsed:.2f}s | Mode: {mode}",
        "sources_label": "📚 Cited Sources ({count} passages)",
        "page_label": "Page",
        "rerank_score": "Rerank Score",
        "cosine_score": "Cosine Score",
        "file_label": "File",
    },
}

# --- SIDEBAR ---
with st.sidebar:
  lang_choice = st.radio(
      "🌐 Ngôn ngữ / Language",
      options=["Tiếng Việt (VN)", "English (EN)"],
      horizontal=True,
  )
  lang = "vi" if "VN" in lang_choice else "en"
  t = TEXTS[lang]

  st.divider()
  st.header(t["sidebar_config"])
  use_reranker = st.toggle(t["use_reranker"], value=True)
  top_k = st.slider(t["top_k"], min_value=1, max_value=5, value=3)

  st.divider()
  st.subheader(t["upload_header"])
  uploaded_files = st.file_uploader(
      t["upload_label"], type=["pdf"], accept_multiple_files=True
  )

  if uploaded_files and st.button(t["index_btn"]):
    with st.spinner(t["indexing_spinner"]):
      new_chunks = []
      for up_file in uploaded_files:
        doc = pymupdf.open(stream=up_file.read(), filetype="pdf")
        for page_num in range(len(doc)):
          clean_text = " ".join(
              doc[page_num].get_text("text").replace("-\n", "").split()
          )
          for c_text in chunk_text(clean_text, chunk_size=250, overlap=50):
            new_chunks.append({
                "chunk_id": len(rag.retriever.chunks) + len(new_chunks),
                "source": up_file.name,
                "page": page_num + 1,
                "text": c_text,
            })
        doc.close()

      if new_chunks:
        embeddings = rag.retriever.embedder.encode(
            [c["text"] for c in new_chunks], normalize_embeddings=True
        )
        rag.retriever.index.add(np.array(embeddings, dtype="float32"))
        rag.retriever.chunks.extend(new_chunks)
        st.success(
            t["indexing_success"].format(
                chunks=len(new_chunks), files=len(uploaded_files)
            )
        )

  st.divider()
  st.caption(t["total_chunks"].format(count=len(rag.retriever.chunks)))
  if st.button(t["clear_btn"]):
    st.session_state.messages = []
    st.rerun()

# --- GIAO DIỆN CHÍNH ---
st.title(t["title"])
st.caption(t["subtitle"])

with st.expander(t["sample_expander"]):
  if os.path.exists("data/test_50_questions.json"):
    with open("data/test_50_questions.json", "r", encoding="utf-8") as f:
      samples = json.load(f)[:5]
    for s in samples:
      st.markdown(
          f"- **{s['question']}** *({t['file_label']}: `{s['pdf_file']}`)*"
      )

if "messages" not in st.session_state:
  st.session_state.messages = []

for msg in st.session_state.messages:
  with st.chat_message(msg["role"]):
    st.markdown(msg["content"])
    if msg.get("chunks"):
      with st.expander(t["sources_label"].format(count=len(msg["chunks"]))):
        for idx, c in enumerate(msg["chunks"], 1):
          score_info = (
              f"{t['rerank_score']}: `{c['rerank_score']:.4f}`"
              if "rerank_score" in c
              else f"{t['cosine_score']}: `{c['retrieval_score']:.4f}`"
          )
          st.markdown(
              f"**[{idx}] {c['source']} — {t['page_label']} {c['page']}**"
              f" ({score_info})"
          )
          st.info(c["text"])

if prompt := st.chat_input(t["chat_placeholder"]):
  st.session_state.messages.append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)

  with st.chat_message("assistant"):
    mode_label = t["mode_main"] if use_reranker else t["mode_base"]
    with st.spinner(t["thinking"].format(mode=mode_label)):
      start_t = time.time()
      result = rag.ask(
          prompt, top_k=top_k, use_reranker=use_reranker, lang=lang
      )
      elapsed = time.time() - start_t

    answer_text = result["answer"]
    st.markdown(answer_text)
    st.caption(t["time_caption"].format(elapsed=elapsed, mode=mode_label))

    with st.expander(t["sources_label"].format(count=len(result["chunks"]))):
      for idx, c in enumerate(result["chunks"], 1):
        score_info = (
            f"{t['rerank_score']}: `{c['rerank_score']:.4f}`"
            if "rerank_score" in c
            else f"{t['cosine_score']}: `{c['retrieval_score']:.4f}`"
        )
        st.markdown(
            f"**[{idx}] {c['source']} — {t['page_label']} {c['page']}**"
            f" ({score_info})"
        )
        st.info(c["text"])

  st.session_state.messages.append({
      "role": "assistant",
      "content": answer_text,
      "chunks": result["chunks"],
  })