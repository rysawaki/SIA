import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAwareEncoder(nn.Module):
    """
    LLaMAから Self-space 入力 u_t を抽出するための Encoder。
    """

    def __init__(self, llama_model, tokenizer, hidden_dim, self_dim):  # <--- tokenizer引数を追加
        super().__init__()
        self.llama = llama_model
        self.tokenizer = tokenizer  # <--- トークナイザーを保持
        self.hidden_dim = hidden_dim
        self.self_dim = self_dim

        # Self-space への写像
        self.project_to_self = nn.Linear(hidden_dim, self_dim)

        # Self-center を LLaMA空間に写す
        self.self_to_llama = nn.Linear(self_dim, hidden_dim)

    def forward(self, input_text: str, self_center: torch.Tensor):
        """
        input_text : str
        self_center : (self_dim,)
        return : u_t (self_dim,)
        """

        # 1) Tokenize (ここでテキストをTensorに変換)
        # モデルと同じデバイスに転送する
        device = next(self.llama.parameters()).device
        inputs = self.tokenizer(input_text, return_tensors="pt").to(device)

        # 2) LLaMA hidden states 取得
        # input_idsなどを **inputs で展開して渡す
        outputs = self.llama(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
        hidden = hidden.squeeze(0)  # (seq_len, hidden_dim)

        # 3) Self-center を LLaMA埋め込み空間へ
        self_proj = self.self_to_llama(self_center)  # (hidden_dim,)

        # 4) tokenごとに自己関係性スコアを計算
        # self_proj と hidden の内積を取る
        scores = torch.matmul(hidden, self_proj)  # (seq_len,)

        # 温度 0.1 でソフトマックス（Selfに近いトークンを強く抽出）
        weights = F.softmax(scores / 0.1, dim=0)

        # 加重平均でプーリング
        pooled = torch.sum(weights.unsqueeze(-1) * hidden, dim=0)

        # 5) Self-space へ写像
        u_t = self.project_to_self(pooled)  # (self_dim,)

        return u_t