import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


class RAGGenerator:

  def __init__(self, model_name: str = LLM_MODEL):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if self.device == "cuda" else torch.float32

    print(f"Đang nạp Open LLM '{model_name}' trên thiết bị: {self.device}...")
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
    ).to(self.device)
    self.model.eval()

  def generate(self, query: str, retrieved_chunks: list, lang: str = "en"):
    if not retrieved_chunks:
      msg = (
          "Tôi không tìm thấy câu trả lời trong các tài liệu được cung cấp."
          if lang == "vi"
          else "I cannot find the answer in the provided documents."
      )
      return msg, []

    context_blocks = []
    sources = []
    seen_sources = set()

    for idx, chunk in enumerate(retrieved_chunks, start=1):
      src_label = f"{chunk['source']} (Page {chunk['page']})"
      context_blocks.append(f"[{idx}] Source: {src_label}\n{chunk['text']}")

      if src_label not in seen_sources:
        seen_sources.add(src_label)
        sources.append({
            "source": chunk["source"],
            "page": chunk["page"],
            "snippet": chunk["text"][:200] + "...",
        })

    context_text = "\n\n".join(context_blocks)

    if lang == "vi":
      system_prompt = (
          "Bạn là một trợ lý nghiên cứu tài liệu học thuật chính xác. Hãy trả"
          " lời câu hỏi của người dùng bằng tiếng Việt một cách ngắn gọn và"
          " chính xác, CHỈ dựa trên các đoạn ngữ cảnh được cung cấp. Không sử"
          " dụng kiến thức bên ngoài hoặc suy đoán. Nếu câu trả lời không có"
          " trong ngữ cảnh, hãy trả lời chính xác câu sau: 'Tôi không tìm thấy"
          " câu trả lời trong các tài liệu được cung cấp.'"
      )
      user_prompt = (
          f"Ngữ cảnh:\n{context_text}\n\nCâu hỏi: {query}\n\nCâu trả lời ngắn"
          " gọn bằng tiếng Việt:"
      )
    else:
      system_prompt = (
          "You are a precise academic document assistant. Answer the user's"
          " question concisely and accurately using ONLY the provided context"
          " excerpts. Do not use outside knowledge or guess. If the answer is"
          " not explicitly stated in the context, reply exactly: 'I cannot find"
          " the answer in the provided documents.'"
      )
      user_prompt = (
          f"Context:\n{context_text}\n\nQuestion: {query}\n\nConcise Answer:"
      )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    text_input = self.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = self.tokenizer([text_input], return_tensors="pt").to(self.device)

    with torch.no_grad():
      generated_ids = self.model.generate(
          **inputs,
          max_new_tokens=180,
          do_sample=False,
          pad_token_id=self.tokenizer.eos_token_id,
      )

    output_ids = generated_ids[0][inputs.input_ids.shape[1] :]
    answer = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    return answer, sources