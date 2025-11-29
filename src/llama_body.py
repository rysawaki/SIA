import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # ★追加: 4-bit 量子化設定 (VRAMを極限まで節約)
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # Normal Float 4-bit
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        # モデルとトークナイザーのロード
        # 既存の float16, device_map="auto" は削除し、4-bit設定に置き換えます
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # torch_dtype=torch.float16,  # ← 削除またはコメントアウト
            # device_map="auto",         # ← 削除またはコメントアウト
            quantization_config=nf4_config,  # ★追加: 4-bit設定を適用
            device_map="auto",  # ★再度追加: device_map="auto" は必須
            low_cpu_mem_usage=True
        )
        self.model.eval()


        # 隠れ層の次元を取得 (TinyLlamaなら2048, Llama-3-8Bなら4096)
        self.hidden_dim = self.model.config.hidden_size
        print(f"Hidden Dimension: {self.hidden_dim}")

        # 予測ヘッドの追加 (Selfの状態から次の埋め込みを予測)
        # Selfの状態ベクトルを受け取り、hidden_dimのベクトルを出力する最小の構造
        # ここではSelfSpaceの次元（hidden_dim）と同じと仮定
        self.prediction_head = nn.Linear(self.hidden_dim, self.hidden_dim).to(self.device).to(torch.float16)


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

    def memorize_experience_vec(self, trace_vec: torch.Tensor, shock: float, affect: float):
        """
        Prediction Error (ベクトル) をTraceとしてSelfに刻む（新ロジック）
        Active Inferenceにおける予測誤差の刻印に使用される。
        """
        self.self_space.update(
            trace=trace_vec.to(self.device).to(torch.float16),
            shock=shock,
            affect=affect
        )
        # ログはController側で行うため、ここでは省略

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

    def generate_with_self_and_get_embed(self, prompt, max_new_tokens=50, alpha=2.0, repetition_penalty=1.2):
        """
        Selfで歪ませた埋め込みで生成を行い、結果テキストと応答の平均埋め込みを返す。
        Active Inferenceにおける「観測（Observed）」を取得するために使用される。
        
        Returns:
            tuple: (generated_text: str, observed_embed: torch.Tensor)
                - generated_text: 生成されたテキスト
                - observed_embed: 応答部分の平均埋め込みベクトル (D,)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # 1. 通常のEmbedding取得
            inputs_embeds = self.model.get_input_embeddings()(inputs.input_ids)  # (B, Seq, Dim)

            # 2. SIAによる歪み (Semantic Gravity)
            B, S, D = inputs_embeds.shape
            flat_embeds = inputs_embeds.view(B * S, D)
            distorted_flat = self.self_space.condition(flat_embeds, alpha=alpha)
            inputs_embeds_distorted = distorted_flat.view(B, S, D)

            # 3. 歪んだ埋め込みを使って生成
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds_distorted,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # 4. 生成部分の埋め込みを取得 (Observed Embed)
            # 全出力の埋め込みを取得
            output_embeds = self.model.get_input_embeddings()(outputs)  # (B, Seq_out, Dim)
            
            # プロンプト部分を除いた「応答」部分の埋め込みのみを抽出
            # inputs.input_ids.shape[-1] がプロンプトの長さ
            prompt_len = inputs.input_ids.shape[-1]
            response_embeds = output_embeds[:, prompt_len:]
            
            # 応答埋め込みの平均を Observed Embed とする (これが観測結果)
            if response_embeds.shape[1] > 0:
                observed_embed = response_embeds.mean(dim=1).squeeze(0)  # (D,)
            else:
                # 応答が空の場合（稀）、プロンプトの最後の埋め込みを使用
                observed_embed = inputs_embeds[:, -1, :].squeeze(0)  # (D,)

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # テキストと埋め込みをセットで返す
            return generated_text, observed_embed

    @torch.no_grad()
    def get_self_state_vector(self):
        """現在のSelfSpaceの状態を単一ベクトルとして圧縮して返す（Predictorの入力用）"""
        k = self.self_space.num_active.item()
        if k == 0:
            return torch.zeros(self.hidden_dim, device=self.device, dtype=torch.float16)

        # 最もシンプルに、有効な軸の加重平均をSelfの代表ベクトルとする
        axes = self.self_space.axes[:k]
        strength = F.relu(self.self_space.strength[:k]) + 1e-6
        weights = strength / strength.sum()
        return torch.matmul(weights.unsqueeze(0), axes).squeeze(0) # (D,)

    def predict_expected_embed(self):
        """Prediction Headを使って、次のステップで期待される埋め込みを生成"""
        state_vec = self.get_self_state_vector()
        # Prediction Headを呼び出す
        return self.prediction_head(state_vec)

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