import json
import os
import tarfile
import time
import urllib.request

RAW_DIR = "data/qasper_raw"
PDF_DIR = "data/pdfs"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

URLS = {
    "train_dev": (
        "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
    ),
    "test": (
        "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz"
    ),
}


def download_and_extract():
  for name, url in URLS.items():
    tgz_path = os.path.join(RAW_DIR, f"{name}.tgz")
    if not os.path.exists(tgz_path):
      print(f"Đang tải gói dữ liệu {name} từ AllenAI...")
      urllib.request.urlretrieve(url, tgz_path)
    with tarfile.open(tgz_path, "r:gz") as tar:
      for member in tar.getmembers():
        if member.name.endswith(".json"):
          member.name = os.path.basename(member.name)
          tar.extract(member, path=RAW_DIR)


download_and_extract()

# Thống kê số lượng bài báo trong từng tập train / validation / test
for split_file in [
    "qasper-train-v0.3.json",
    "qasper-dev-v0.3.json",
    "qasper-test-v0.3.json",
]:
  path = os.path.join(RAW_DIR, split_file)
  if os.path.exists(path):
    with open(path, "r", encoding="utf-8") as f:
      d = json.load(f)
    print(f"Tập {split_file}: {len(d)} bài báo")

with open(
    os.path.join(RAW_DIR, "qasper-test-v0.3.json"), "r", encoding="utf-8"
) as f:
  test_data = json.load(f)

test_questions = []
downloaded_pdfs = 0

for paper_id, paper in test_data.items():
  if len(test_questions) >= 50:
    break

  # Lọc các câu hỏi có đáp án chuẩn và đoạn văn bằng chứng (evidence)
  valid_qas = []
  for qa in paper.get("qas", []):
    answers = qa.get("answers", [])
    if not answers:
      continue
    first_ans = answers[0].get("answer", {})
    if first_ans.get("unanswerable", True):
      continue

    gold_answer = first_ans.get("free_form_answer", "").strip()
    if not gold_answer and first_ans.get("extractive_spans"):
      gold_answer = " ".join(first_ans["extractive_spans"]).strip()
    if not gold_answer and first_ans.get("yes_no") is not None:
      gold_answer = "Yes" if first_ans["yes_no"] else "No"

    evidence_list = [
        ev.strip() for ev in first_ans.get("evidence", []) if ev.strip()
    ]

    if gold_answer and evidence_list:
      valid_qas.append({
          "paper_id": paper_id,
          "pdf_file": f"{paper_id.replace('/', '_')}.pdf",
          "question": qa["question"],
          "gold_answer": gold_answer,
          "evidence": evidence_list,
      })

  if not valid_qas:
    continue

  # Tải file PDF gốc từ arXiv
  safe_filename = f"{paper_id.replace('/', '_')}.pdf"
  pdf_path = os.path.join(PDF_DIR, safe_filename)
  pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"

  if not os.path.exists(pdf_path):
    try:
      req = urllib.request.Request(
          pdf_url, headers={"User-Agent": "Mozilla/5.0"}
      )
      with urllib.request.urlopen(req, timeout=20) as resp, open(
          pdf_path, "wb"
      ) as f:
        f.write(resp.read())
      downloaded_pdfs += 1
      print(f"Đã tải PDF: {safe_filename}")
      time.sleep(1)
    except Exception as e:
      print(f"Bỏ qua {paper_id} do lỗi tải PDF: {e}")
      continue

  for item in valid_qas:
    if len(test_questions) >= 50:
      break
    test_questions.append(item)

with open("data/test_50_questions.json", "w", encoding="utf-8") as f:
  json.dump(test_questions, f, ensure_ascii=False, indent=2)

print(
    f"\nHoàn tất! Đã tải {downloaded_pdfs} file PDF và lưu"
    f" {len(test_questions)} câu hỏi test."
)