# Document Chat with Retrieval-Augmented Generation (RAG)

Hệ thống hỏi-đáp trên bộ sưu tập tài liệu PDF sử dụng kiến trúc **Retrieval-Augmented Generation (RAG)** kết hợp **Dense Embedding Retrieval** và **Cross-Encoder Reranking** với mô hình ngôn ngữ mở (**Open LLM**).

## 1. Kiến trúc Hệ thống & Mô hình

* **PDF Parser & Chunking:** `PyMuPDF` trích xuất văn bản theo từng trang, làm sạch ngắt dòng và chia đoạn kích thước `250 words` với độ gối đầu `overlap = 50 words` (tổng cộng 476 chunks từ 16 bài báo).
* **Embedding Model:** `BAAI/bge-small-en-v1.5` (384 chiều, chuẩn hóa L2) lưu trữ trên cơ sở dữ liệu vector **FAISS** (`IndexFlatIP`).
* **Reranker (Main Model):** `cross-encoder/ms-marco-MiniLM-L6-v2` chấm điểm tương tác chéo trên Top-20 ứng viên từ FAISS để chọn lại Top-$k$ đoạn văn chính xác nhất.
* **Open LLM Generator:** `Qwen/Qwen2.5-1.5B-Instruct` sinh câu trả lời súc tích dựa trên ngữ cảnh được cấp, kèm trích dẫn tên file PDF và số trang.
* **Hallucination Evaluator:** `cross-encoder/nli-deberta-v3-small` kiểm tra tính suy diễn trung thực (Entailment) giữa câu trả lời và từng đoạn ngữ cảnh.
* **Chat UI:** Xây dựng bằng `Streamlit`, hỗ trợ bật/tắt Reranker trực tiếp, điều chỉnh `Top-k` và tải thêm tài liệu PDF mới.

## 2. Bộ Dữ liệu (Dataset)

* Sử dụng bộ dữ liệu khoa học công khai **QASPER** (`allenai/qasper` v0.3) với phân chia chuẩn `train` (888 bài báo), `validation/dev` (281 bài báo), và `test` (416 bài báo).
* Trích xuất **16 bài báo PDF** từ tập `test` trên arXiv kèm **50 câu hỏi kiểm thử** (`data/test_50_questions.json`) có sẵn đáp án chuẩn (`gold_answer`) và đoạn văn bằng chứng (`evidence`).

## 3. Hướng dẫn Cài đặt & Chạy Dự án

```bash
# 1. Tạo và kích hoạt môi trường ảo
python -m venv venv
venv\Scripts\activate  # Trên Linux/macOS: source venv/bin/activate

# 2. Cài đặt thư viện
pip install -r requirements.txt

# 3. Tải dữ liệu QASPER & PDF
python download_data.py

# 4. Tạo FAISS Vector Index
python src/ingest.py

# 5. Chạy đánh giá thực nghiệm trên 50 câu hỏi test & xuất biểu đồ
python -m eval.evaluate
python -m eval.analyze_results

# 6. Khởi chạy giao diện Chat UI
streamlit run app.py
