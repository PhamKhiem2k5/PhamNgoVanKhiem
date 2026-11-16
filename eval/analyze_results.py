import json
import re
import string
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder
from src.retriever import RAGRetriever

CSV_PATH = "eval/results_comparison.csv"
TEST_FILE = "data/test_50_questions.json"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
CHART_PATH = "eval/metrics_comparison.png"


def normalize_text(text: str) -> str:
  text = text.lower()
  text = "".join(ch for ch in text if ch not in set(string.punctuation))
  text = re.sub(r"\b(a|an|the)\b", " ", text)
  return " ".join(text.split())


def check_recall_at_k(
    retrieved_chunks: list,
    target_pdf: str,
    evidence_list: list,
    threshold: float = 0.45,
) -> int:
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
      if len(chunk_tokens & ev_tokens) / len(ev_tokens) >= threshold:
        return 1
  return 0


def check_hallucination_per_chunk(
    answer: str, chunks: list, nli: CrossEncoder, id2label: dict
) -> int:
  if not isinstance(answer, str) or "cannot find the answer" in answer.lower():
    return 0
  pairs = [(c["text"], answer) for c in chunks]
  scores = nli.predict(pairs)
  for row in scores:
    label = id2label[int(np.argmax(row))].lower()
    if "entail" in label:
      return 0
  return 1


def main():
  df = pd.read_csv(CSV_PATH)
  with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_data = json.load(f)

  retriever = RAGRetriever()
  nli = CrossEncoder(NLI_MODEL)
  id2label = nli.model.config.id2label

  base_halluc_fixed, main_halluc_fixed = [], []
  base_recall_5, main_recall_5 = [], []

  for idx, row in df.iterrows():
    q = row["question"]
    target_pdf = test_data[idx]["pdf_file"]
    evidence = test_data[idx]["evidence"]

    # Truy xuất Top-5 để tính Recall@5, và dùng Top-3 đầu tiên để khớp với ngữ cảnh LLM đã đọc
    base_chunks_5 = retriever.retrieve(q, top_k=5, use_reranker=False)
    main_chunks_5 = retriever.retrieve(q, top_k=5, use_reranker=True)

    base_recall_5.append(check_recall_at_k(base_chunks_5, target_pdf, evidence))
    main_recall_5.append(check_recall_at_k(main_chunks_5, target_pdf, evidence))

    base_halluc_fixed.append(
        check_hallucination_per_chunk(
            row["base_answer"], base_chunks_5[:3], nli, id2label
        )
    )
    main_halluc_fixed.append(
        check_hallucination_per_chunk(
            row["main_answer"], main_chunks_5[:3], nli, id2label
        )
    )

  df["base_recall@5"] = base_recall_5
  df["main_recall@5"] = main_recall_5
  df["base_hallucination"] = base_halluc_fixed
  df["main_hallucination"] = main_halluc_fixed
  df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

  base_scores = [
      round(df["base_recall@3"].mean() * 100, 2),
      round(df["base_recall@5"].mean() * 100, 2),
      round(df["base_f1"].mean() * 100, 2),
      round(df["base_hallucination"].mean() * 100, 2),
  ]
  main_scores = [
      round(df["main_recall@3"].mean() * 100, 2),
      round(df["main_recall@5"].mean() * 100, 2),
      round(df["main_f1"].mean() * 100, 2),
      round(df["main_hallucination"].mean() * 100, 2),
  ]

  metrics = [
      "Recall@3 (%)",
      "Recall@5 (%)",
      "Answer F1 (%)",
      "Hallucination Rate (%)",
  ]
  summary = pd.DataFrame({
      "Metric": metrics,
      "Baseline (No Reranker)": base_scores,
      "Main Model (With Reranker)": main_scores,
  })

  print("\n================ BẢNG KẾT QUẢ CHÍNH THỨC ================")
  print(summary.to_string(index=False))

  # Vẽ biểu đồ so sánh nếu đã cài matplotlib
  try:
    import matplotlib.pyplot as plt

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.8))
    rects1 = ax.bar(
        x - width / 2,
        base_scores,
        width,
        label="Baseline (No Reranker)",
        color="#64748b",
    )
    rects2 = ax.bar(
        x + width / 2,
        main_scores,
        width,
        label="Main Model (With Reranker)",
        color="#2563eb",
    )

    ax.set_ylabel("Percentage (%)")
    ax.set_title("RAG Performance Comparison on 50 QASPER Test Questions")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, max(max(base_scores), max(main_scores)) + 15)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for rect in rects1 + rects2:
      h = rect.get_height()
      ax.annotate(
          f"{h:.1f}%",
          xy=(rect.get_x() + rect.get_width() / 2, h),
          xytext=(0, 3),
          textcoords="offset points",
          ha="center",
          va="bottom",
          fontsize=9,
      )

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=300)
    print(f"\nĐã xuất biểu đồ báo cáo tại: {CHART_PATH}")
  except ImportError:
    print(
        "\n(Gợi ý: Chạy 'pip install matplotlib' rồi chạy lại script này để"
        " tự động xuất ảnh biểu đồ.)"
    )

  print("\n--- VÍ DỤ 1: RERANKER CỨU THÀNH CÔNG (Baseline sai, Main đúng) ---")
  wins = df[(df["base_recall@3"] == 0) & (df["main_recall@3"] == 1)].head(2)
  for _, r in wins.iterrows():
    print(
        f"• Q: {r['question']}\n  Gold: {r['gold_answer']}\n  Base:"
        f" {r['base_answer']}\n  Main: {r['main_answer']}\n"
    )

  print("--- VÍ DỤ 2: CÁC CA THẤT BẠI (Dùng cho Error Analysis) ---")
  fails = df[(df["main_recall@3"] == 0) | (df["main_f1"] == 0)].head(3)
  for _, r in fails.iterrows():
    print(
        f"• Q: {r['question']}\n  Gold: {r['gold_answer']}\n  Main:"
        f" {r['main_answer']} (Recall@3={r['main_recall@3']},"
        f" F1={r['main_f1']})\n"
    )


if __name__ == "__main__":
  main()