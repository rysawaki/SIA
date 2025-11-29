import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os

# パス解決（既存のディレクトリ構造に合わせています）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.self_space import SelfSpace


class SelfInjectedLlama(nn.Module):
    """
    Llamaの入力埋め込み層にSIA (Self-Imprint) を介入させるラッパー。
    入力テキスト -> Embedding -> [SIAによる歪み] -> Llama Layers -> 生成
    """

    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device="cuda"):
        super().__init__()
        print(f"Loading Llama model: {model_name}...")
        self.device = device

        # モデルとトークナイザーのロード
        # GTX 1650 (4GB) 向けに float16 と device_map="auto" を使用
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.model.eval()

        # 隠れ層の次元を取得 (TinyLlamaなら2048, Llama-3-8Bなら4096)
        self.hidden_dim = self.model.config.hidden_size
        print(f"Hidden Dimension: {self.hidden_dim}")


        # SelfSpaceの初期化 (モデルの次元に合わせる)
        self.self_space = SelfSpace(dim=self.hidden_dim, max_axes=5, device=device)

        # ★追加: SelfSpaceの全パラメータを明示的にGPUへ送る
        self.self_space.to(self.device)

        self.self_space.to(torch.float16)  # 型合わせ

    def memorize_experience(self, text, shock=1.0, affect=1.0):
        """
        経験をテキストとして入力し、そのEmbeddingをTraceとしてSelfに刻む
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # Embedding層だけを通す
            embeds = self.model.get_input_embeddings()(inputs.input_ids)
            # 文全体の平均ベクトルをTraceとする
            trace = embeds.mean(dim=1).squeeze(0)

        self.self_space.update(trace, shock=shock, affect=affect)
        print(f"Experience Memorized: '{text}' (Shock={shock})")

    def generate_with_self(self, prompt, max_new_tokens=50, alpha=2.0, repetition_penalty=1.2):
        """
        Promptの埋め込みをSelfSpaceで歪ませてから生成を行う
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # 1. 通常のEmbedding取得
            inputs_embeds = self.model.get_input_embeddings()(inputs.input_ids)  # (B, Seq, Dim)

            # 2. SIAによる歪み (Semantic Gravity)
            # バッチサイズ次元を維持しつつ適用
            B, S, D = inputs_embeds.shape
            flat_embeds = inputs_embeds.view(B * S, D)

            # SelfSpace.conditionはベクトルを歪ませる
            distorted_flat = self.self_space.condition(flat_embeds, alpha=alpha)
            inputs_embeds_distorted = distorted_flat.view(B, S, D)

            # 3. 歪んだ埋め込みを使って生成
            # generateメソッドは inputs_embeds を受け取れる
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds_distorted,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # inputs_embedsを使った場合、生成部分のみをデコードするのが理想だが
            # ここでは簡易的に全出力をデコードする
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text


def run_llama_experiment():
    # 軽量モデルを指定 (VRAM 4GB対応)
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    agent = SelfInjectedLlama(model_name=model_name, device=device)

    # テスト用プロンプト
    prompt = "Context: A new friend approaches you with a smile.\nYou say:"

    print("\n=== 1. Baseline Generation (Pure Llama) ===")
    # Selfが空の状態（alphaが効かない）での生成
    print(agent.generate_with_self(prompt, alpha=0.0))

    print("\n=== 2. Experiencing Trauma ===")
    # 強いトラウマを刻む
    # 「笑顔の友人に裏切られた」「信頼は痛みだ」という経験
    agent.memorize_experience("I was betrayed by my best friend who smiled at me.", shock=1.0)
    agent.memorize_experience("Trust leads to pain. Everyone lies.", shock=1.0)

    print(f"Self Metrics: {agent.self_space.metrics()}")

    print("\n=== 3. Distorted Generation (With Self) ===")
    # 強い歪み (alpha=5.0) をかけて生成
    # 埋め込み空間での移動距離を稼ぐため、少し強めのalphaに設定
    output = agent.generate_with_self(prompt, alpha=5.0)

    print("-" * 50)
    print(f"Prompt: {prompt}")
    print(f"Output: {output}")
    print("-" * 50)

    print("\n[Check]")
    print("Does the output reflect distrust or hesitation?")


    print("\n=== 4. Finding the Sweet Spot (Alpha Tuning) ===")

    # 刻んでテストする
    for a in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        print(f"\n--- Alpha = {a} ---")
        try:
            output = agent.generate_with_self(prompt, alpha=a)
            # プロンプトの繰り返しを避けて、応答部分だけ見やすく表示
            clean_output = output.replace(prompt, "").strip()
            print(f"Output: {clean_output}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    run_llama_experiment()