from collections import Counter
import json
import re
import string
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder
from tqdm import tqdm
from src.pipeline import RAGPipeline

TEST_FILE = "data/test_50_questions.json"
OUTPUT_CSV = "eval/results_comparison.csv"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"


def normalize_text(text: str) -> str:
  """Chuẩn hóa văn bản theo chuẩn SQuAD để tính Answer F1."""
  text = text.lower()
  text = "".join(ch for ch in text if ch not in set(string.punctuation))
  text = re.sub(r"\b(a|an|the)\b", " ", text)
  return " ".join(text.split())


def compute_f1(prediction: str, ground_truth: str) -> float:
  pred_tokens = normalize_text(prediction).split()
  gold_tokens = normalize_text(ground_truth).split()
  if not pred_tokens or not gold_tokens:
    return float(pred_tokens == gold_tokens)

  common = Counter(pred_tokens) & Counter(gold_tokens)
  num_same = sum(common.values())
  if num_same == 0:
    return 0.0

  precision = num_same / len(pred_tokens)
  recall = num_same / len(gold_tokens)
  return (2 * precision * recall) / (precision + recall)


def check_recall_at_k(
    retrieved_chunks: list,
    target_pdf: str,
    evidence_list: list,
    threshold: float = 0.45,
) -> int:
  """Trả về 1 nếu ít nhất một đoạn evidence chuẩn xuất hiện trong Top-k chunks của đúng PDF."""
  for chunk in retrieved_chunks:
    if chunk["source"] != target_pdf:
      continue
    chunk_tokens = set(normalize_text(chunk["text"]).split())
    if not chunk_tokens:
      continue

    for ev in evidence_list:
      ev_tokens = set(normalize_text(ev).split())
      if not ev_tokens:
        continue
      overlap_ratio = len(chunk_tokens & ev_tokens) / len(ev_tokens)
      if overlap_ratio >= threshold:
        return 1
  return 0


class HallucinationChecker:

  def __init__(self, model_name: str = NLI_MODEL):
    print(f"Đang nạp mô hình kiểm tra Hallucination ({model_name})...")
    self.nli = CrossEncoder(model_name)
    self.id2label = self.nli.model.config.id2label

  def is_hallucinated(self, answer: str, retrieved_chunks: list) -> int:
    """Trả về 1 nếu câu trả lời không được hỗ trợ bởi context (hallucination), 0 nếu trung thực."""
    if "cannot find the answer" in answer.lower():
      return 0  # Từ chối trả lời khi thiếu thông tin không bị coi là bịa đặt

    context = " ".join([c["text"] for c in retrieved_chunks])
    scores = self.nli.predict([(context, answer)])[0]
    pred_id = int(np.argmax(scores))
    label = self.id2label[pred_id].lower()

    # Nếu nhãn NLI là 'entailment' -> trung thực (0); 'contradiction' hoặc 'neutral' -> ảo giác (1)
    return 0 if "entail" in label else 1


def run_evaluation():
  with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_data = json.load(f)

  rag = RAGPipeline()
  hallucination_checker = HallucinationChecker()

  records = []
  print(
      f"\nBắt đầu đánh giá trên {len(test_data)} câu hỏi (Baseline vs Main"
      " Model)..."
  )

  for idx, item in enumerate(tqdm(test_data), start=1):
    q = item["question"]
    gold = item["gold_answer"]
    target_pdf = item["pdf_file"]
    evidence = item["evidence"]

    # 1. Chạy Baseline (Không Reranker)
    res_base = rag.ask(q, top_k=3, use_reranker=False)
    base_recall = check_recall_at_k(res_base["chunks"], target_pdf, evidence)
    base_f1 = compute_f1(res_base["answer"], gold)
    base_halluc = hallucination_checker.is_hallucinated(
        res_base["answer"], res_base["chunks"]
    )

    # 2. Chạy Main Model (Có Reranker)
    res_main = rag.ask(q, top_k=3, use_reranker=True)
    main_recall = check_recall_at_k(res_main["chunks"], target_pdf, evidence)
    main_f1 = compute_f1(res_main["answer"], gold)
    main_halluc = hallucination_checker.is_hallucinated(
        res_main["answer"], res_main["chunks"]
    )

    records.append({
        "id": idx,
        "pdf_file": target_pdf,
        "question": q,
        "gold_answer": gold,
        "base_answer": res_base["answer"],
        "base_recall@3": base_recall,
        "base_f1": round(base_f1, 4),
        "base_hallucination": base_halluc,
        "main_answer": res_main["answer"],
        "main_recall@3": main_recall,
        "main_f1": round(main_f1, 4),
        "main_hallucination": main_halluc,
    })

  df = pd.DataFrame(records)
  df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

  print("\n================ KẾT QUẢ TỔNG HỢP (50 CÂU HỎI) ================")
  summary = pd.DataFrame({
      "Metric": ["Recall@3 (%)", "Answer F1 (%)", "Hallucination Rate (%)"],
      "Baseline (No Reranker)": [
          round(df["base_recall@3"].mean() * 100, 2),
          round(df["base_f1"].mean() * 100, 2),
          round(df["base_hallucination"].mean() * 100, 2),
      ],
      "Main Model (With Reranker)": [
          round(df["main_recall@3"].mean() * 100, 2),
          round(df["main_f1"].mean() * 100, 2),
          round(df["main_hallucination"].mean() * 100, 2),
      ],
  })
  print(summary.to_string(index=False))
  print(f"\nĐã lưu kết quả chi tiết từng câu hỏi tại: {OUTPUT_CSV}")


if __name__ == "__main__":
  run_evaluation()