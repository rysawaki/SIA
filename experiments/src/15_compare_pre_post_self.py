# ==============================================
# compare_pre_post_self.py
# --- Experiment:
# Same prompt, but generation changes
# before and after Self evolution
# ==============================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.visualize_selfspace import SelfSpace
from trace_extractor import TraceExtractor
from models.transformer_block_sia import SIATransformerBlock


# ====== 設定 ======
MODEL_NAME = "gpt2"            # HuggingFaceモデル (軽量でOK)
device = "cuda" if torch.cuda.is_available() else "cpu"
dim = 768                      # GPT-2 hidden size


# ====== モデル準備 ======
print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# Self-SIA Transformer block (1層だけ置き換える簡易版)
sia_block = SIATransformerBlock(dim=dim, num_heads=12, device=device).to(device)

self_space = sia_block.self_space
extractor = TraceExtractor(device=device)


# ====== Utility: 文をLLMで生成 ======
def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = base_model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )
    return tokenizer.decode(outputs[0])


# ====== ① Self変形前の生成 ======
prompt = "What is love?"
print("\n=== [PRE-SELF] ===")
before = generate(prompt)
print(before)


# ====== ② Trace（経験）を与えてSelfを更新 ======
experience_text = "Love is not just happiness. It often includes pain, loss, and personal growth."

# 入力 & 再構成埋め込み（簡易版：Embedding層を使う）
inputs = tokenizer(experience_text, return_tensors="pt").to(device)
with torch.no_grad():
    input_emb = base_model.transformer.wte(inputs["input_ids"]).mean(dim=1)       # (1,d)
    recon_emb = base_model.transformer.wte(inputs["input_ids"][:, :-1]).mean(dim=1)  # 適当な近似
    input_emb = input_emb.squeeze(0)
    recon_emb = recon_emb.squeeze(0)

# Trace抽出
trace, shock, affect = extractor.extract(
    input_emb=input_emb,
    recon_emb=recon_emb,
    self_state=self_space.self_state
)

print("\n[Trace Injection]")
print(f"shock={shock:.3f}, affect={affect:.3f}")

# Self更新
self_space.update(trace, shock=shock, affect=affect)

print("\n[Self metrics after update]")
print(self_space.metrics())


# ====== ③ Self変形後の生成 ======
print("\n=== [POST-SELF] ===")
after = generate(prompt)
print(after)


# ====== 結果比較 ======
print("\n======================================")
print("PROMPT:", prompt)
print("\n--- Before ---\n", before)
print("\n--- After ---\n", after)
print("======================================")
