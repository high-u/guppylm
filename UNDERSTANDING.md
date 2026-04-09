# GuppyLM を理解する — ソフトウェアエンジニア向け解説

## この文書の目的

GuppyLM のソースコードが「何をしていて」「なぜそれで動くのか」を、LLM を使った開発経験はあるが、モデルの内部構造は詳しくないソフトウェアエンジニア向けに解説する。

---

## 全体像：何が起きているのか

```
[1. データ生成]  →  [2. トークナイザ学習]  →  [3. モデル学習]  →  [4. 推論]
  prepare            prepare                  train              chat
```

LLM の学習とは、大量のテキストから「次の単語を予測する関数」を作ること。
GuppyLM は、この一連の流れを 4 ファイル・約 400 行で実装している。

---

## ステップ 1: データ生成 (`generate_data.py`)

### 何をしているか

テンプレートとランダム要素を組み合わせて、60,000 件の会話データを生成する。

```
入力: "hi guppy"
出力: "hello. the water is clear today. i blew some bubbles earlier."
```

### なぜこれで動くのか

LLM は「こう聞かれたら、こう答える」パターンをテキストから学ぶ。
人間が書く必要はなく、テンプレート × ランダム組み合わせで十分な多様性が出る。

- 30 種の水槽オブジェクト × 20 箇所のスポット × 17 種の食べ物 × 25 種の活動...
- 60 トピック（挨拶、食事、温度、感情、騒音...）ごとにテンプレート関数がある
- `pick()` でランダム選択、`maybe()` で確率的に要素を含めるかどうか決める

### 出力フォーマット

```
<|im_start|>user
hi guppy<|im_end|>
<|im_start|>assistant
hello. the water is clear today.<|im_end|>
```

llama.cpp で見慣れた ChatML フォーマットそのもの。`<|im_start|>` と `<|im_end|>` は特殊トークンとして ID 1, 2 が割り当てられる。

生成結果は `data/train.jsonl`（57,000 件）と `data/eval.jsonl`（3,000 件）に保存される。

---

## ステップ 2: トークナイザ学習 (`prepare_data.py`)

### 何をしているか

生成したテキストから BPE (Byte Pair Encoding) トークナイザを学習する。

### あなたが既に知っていること、との接点

llama.cpp でモデルを動かすとき、プロンプトがトークンに分割されてモデルに渡されることは知っているはず。そのトークン分割器を、ここではゼロから作っている。

### 仕組み

1. テキストをバイト単位に分解する
2. 隣接するバイトペアの出現頻度を数える
3. 最も頻出のペアを 1 つのトークンとして統合する
4. これを語彙サイズ（4,096）に達するまで繰り返す

```python
vocab_size = 4096          # 語彙サイズ。GPT-4 は ~100K、Llama は 32K
special_tokens = [
    "<pad>",         # 0: パディング（短い文を揃える詰め物）
    "<|im_start|>",  # 1: 発話の開始
    "<|im_end|>",    # 2: 発話の終了
]
```

語彙が 4,096 と極端に小さいのは、モデルが小さいから。大きな語彙は表現力が上がるが、それを学習するにはパラメータが必要。9M パラメータに見合った語彙サイズ。

---

## ステップ 3: モデル構造 (`model.py`, `config.py`)

ここが本丸。約 130 行で Transformer が実装されている。

### 設定値の意味

```python
@dataclass
class GuppyConfig:
    vocab_size: int = 4096     # トークナイザの語彙サイズ
    max_seq_len: int = 128     # 一度に処理できるトークン数（コンテキスト長）
    d_model: int = 384         # 各トークンを表すベクトルの次元数
    n_layers: int = 6          # Transformer ブロックの積み重ね数
    n_heads: int = 6           # アテンションヘッドの数
    ffn_hidden: int = 768      # FFN の中間層の次元数（d_model の 2 倍）
    dropout: float = 0.1       # 学習時にランダムにニューロンを無効化する割合
```

llama.cpp で `--ctx-size 4096` と指定するのは、ここでいう `max_seq_len` に相当する。GuppyLM は 128 トークンしか見えない。魚の短期記憶というキャラ設定と一致している。

### モデルの処理フロー

```
入力トークン列: [1, 234, 56, 789, 12, ...]
         │
    ┌────▼────┐
    │ Embedding │  各トークン ID → 384 次元ベクトルに変換
    │ + Position│  + 位置情報を加算（何番目のトークンかを教える）
    └────┬────┘
         │
    ┌────▼────┐
    │  Block ×6 │  Transformer ブロックを 6 回繰り返す
    │ (後述)    │
    └────┬────┘
         │
    ┌────▼────┐
    │ LM Head  │  384 次元ベクトル → 4096 次元（各トークンの確率）に変換
    └────┬────┘
         │
    次のトークンの確率分布
```

### Embedding（埋め込み）

```python
self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)   # 4096 × 384 の行列
self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)  # 128 × 384 の行列
```

- `tok_emb`: トークン ID を 384 次元ベクトルに変換するルックアップテーブル。「hello」と「food」は意味が違うので、異なるベクトルになる。このベクトルの値は学習で決まる。
- `pos_emb`: 位置情報。トークンの並び順を教える。「fish eat food」と「food eat fish」は同じトークン集合だが意味が違うので、位置を教える必要がある。

この 2 つを足し合わせた結果が、Transformer ブロックへの入力になる。

### Transformer Block（`Block` クラス）

```python
def forward(self, x, mask=None):
    x = x + self.attn(self.norm1(x), mask)    # (1) Attention
    x = x + self.ffn(self.norm2(x))           # (2) FFN
    return x
```

各ブロックは 2 つの処理を行い、それぞれの結果を入力に「足す」（残差接続）。

#### (1) Self-Attention（自己注意機構）

「各トークンが、他のどのトークンに注目すべきか」を計算する。

```python
# 入力 x から Q, K, V を作る（1 つの線形変換で 3 つ分を一度に計算）
qkv = self.qkv(x)  # [B, T, 3 * 384] → Q, K, V に分割

# アテンションスコア = Q と K の内積 / sqrt(head_dim)
attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

# マスク適用（未来のトークンを見えなくする）
attn = attn.masked_fill(mask == 0, float("-inf"))

# softmax で正規化 → 重み付き合計
output = attn @ v
```

ソフトウェアエンジニア的に言うと：

- **Q (Query)**: 「私は何を探しているか」 — 各トークンの検索クエリ
- **K (Key)**: 「私は何を持っているか」 — 各トークンのインデックスキー
- **V (Value)**: 「私の中身」 — 各トークンが持つ実際の情報

Q と K の内積で「関連度スコア」を計算し、そのスコアで V を重み付き平均する。
データベースの検索に例えると、Q でインデックス（K）を引き、ヒットした行の値（V）を取り出す操作。ただし完全一致ではなく、類似度に応じた重み付き合計。

**マルチヘッド**: ヘッド数 6 は、6 つの異なる「注目の仕方」を並行して行うこと。あるヘッドは構文的な関係を見て、別のヘッドは意味的な関連を見る...といった役割分担が学習で自然に生まれる。`d_model=384` を `n_heads=6` で割ると、各ヘッドは 64 次元で動作する。

**因果マスク**: `torch.tril` で下三角行列を作り、未来のトークンを見えなくする。トークン 3 はトークン 1, 2, 3 しか見えない。これが「次の単語予測」を可能にする鍵。llama.cpp で KV cache を使うのも、この因果的な構造に基づいている。

#### (2) FFN（フィードフォワードネットワーク）

```python
def forward(self, x):
    return self.dropout(self.down(F.relu(self.up(x))))
    # 384 → 768 → ReLU → 768 → 384
```

Attention が「どのトークンの情報を集めるか」を決めるのに対し、FFN は「集めた情報をどう変換するか」を担う。パターン認識・知識の格納層とも言われる。

- `up`: 384 次元 → 768 次元に拡張
- `ReLU`: 負の値を 0 にする（非線形性の導入）
- `down`: 768 次元 → 384 次元に戻す

### LM Head（出力層）

```python
self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
self.lm_head.weight = self.tok_emb.weight  # Weight Tying
```

最終的な 384 次元ベクトルを 4,096 次元（語彙サイズ）に変換し、各トークンの「次に来る確率」を出力する。

**Weight Tying**: `lm_head` と `tok_emb` の重みを共有する。入力側（トークン → ベクトル）と出力側（ベクトル → トークン確率）で同じ行列を使う。パラメータ数を節約しつつ、入力と出力の表現空間を揃える効果がある。9M パラメータの小さなモデルでは特に効果的。

### パラメータ数の内訳

```
tok_emb:  4096 × 384          = 1,572,864  (lm_head と共有)
pos_emb:  128 × 384           =    49,152
Block ×6:
  norm1:  384 × 2             =       768  (×6)
  qkv:    384 × (384×3)       =   442,368  (×6)
  out:    384 × 384           =   147,456  (×6)
  norm2:  384 × 2             =       768  (×6)
  up:     384 × 768           =   294,912  (×6)
  down:   768 × 384           =   294,912  (×6)
                               ──────────
Block 1 つ:                    ≈ 1,181,184
Block 6 つ:                    ≈ 7,087,104
最終 norm: 384 × 2            =       768
                               ──────────
合計:                           ≈ 8,709,888 (8.7M)
```

---

## ステップ 4: 学習 (`train.py`)

### 何をしているか

「入力テキストの次のトークンを正しく予測できるように、モデルのパラメータを調整する」ことを 10,000 ステップ繰り返す。

### 学習ループの流れ

```python
for step in range(max_steps):      # 10,000 回
    x, y = next(batch)              # x: 入力トークン列, y: 正解（1 つずらしたトークン列）
    logits, loss = model(x, y)      # 予測と損失計算
    loss.backward()                 # 勾配計算（逆伝播）
    optimizer.step()                # パラメータ更新
```

### 入力と正解の関係

```
テキスト:  <|im_start|> user \n hi guppy <|im_end|> <|im_start|> assistant \n hello . <|im_end|>
トークンID: [1,         234,   56, 789,12, 2,        1,          567,     89, 345, 90, 2]

入力 (x):  [1, 234, 56, 789, 12, 2,  1, 567, 89, 345, 90]
正解 (y):  [234, 56, 789, 12, 2,  1, 567, 89, 345, 90,  2]
```

入力を 1 トークンずらしたものが正解。各位置で「次に来るトークン」を当てる問題に変換される。これが `dataset.py` の `x = ids[:-1]`, `y = ids[1:]` の意味。

### 損失関数

```python
loss = F.cross_entropy(logits, targets, ignore_index=0)
```

- モデルの出力（4096 次元の確率分布）と正解トークンの差を計算
- `ignore_index=0`: パディングトークンは無視する（短いシーケンスを揃えるための詰め物なので）

### 学習率スケジューラ

```python
def get_lr(step, config):
    if step < warmup_steps:          # 最初の 200 ステップ: 0 → 3e-4 に線形増加
        return lr * step / warmup_steps
    # 残りは cosine decay: 3e-4 → 3e-5 にゆるやかに減少
    coeff = 0.5 * (1 + cos(pi * progress))
    return min_lr + (lr - min_lr) * coeff
```

最初は小さい学習率で慎重に始め、暖まったら大きく、終盤に向けて小さくする。大きすぎると発散、小さすぎると収束が遅い。

### AMP（Automatic Mixed Precision）

```python
with torch.amp.autocast("cuda"):
    _, loss = model(x, y)
scaler.scale(loss).backward()
```

GPU では FP16（半精度）で計算すると速い。ただし精度が落ちるので、FP32 と自動的に使い分ける。RTX 5070 Ti なら大きな高速化が得られる。

### チェックポイント保存

- `best_model.pt`: 評価損失が最小のモデル（推論で使う）
- `step_N.pt`: N ステップごとのスナップショット
- `final_model.pt`: 最終状態

---

## ステップ 5: 推論 (`inference.py`)

### 何をしているか

学習済みモデルを読み込み、ユーザーの入力に対して 1 トークンずつ生成する。

### 生成の仕組み（`generate` メソッド）

```python
for _ in range(max_new_tokens):        # 最大 64 トークン生成
    logits, _ = self(idx_cond)          # 現在のトークン列を入力
    logits = logits[:, -1, :] / temperature  # 最後の位置の出力だけ使う
    # top_k フィルタリング
    v, _ = torch.topk(logits, top_k)
    logits[logits < v[:, [-1]]] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1)  # 確率的にサンプリング
    idx = torch.cat([idx, next_id], dim=1)  # 生成したトークンを追加
    if next_id.item() == eos_id:        # <|im_end|> が出たら終了
        break
```

llama.cpp の `--temp`, `--top-k` と同じパラメータ。

- **temperature (0.7)**: 確率分布のシャープさ。低いと確定的、高いとランダム
- **top_k (50)**: 上位 50 トークンだけを候補にする。低確率のゴミトークンを排除

llama.cpp との違い: llama.cpp は KV cache を使って過去の計算結果を再利用するが、GuppyLM は毎回全トークンを再計算する。128 トークンと小さいので問題にならない。

### プロンプトのフォーマット

```python
def _format_prompt(self, messages):
    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")  # ← ここから先を生成させる
    return "\n".join(parts)
```

llama-server で `--chat-template chatml` を指定したときと同じ構造。学習データがこのフォーマットで作られているので、推論時も同じフォーマットにする必要がある。

---

## データの流れ：全体を通して

```
"hi guppy" (文字列)
    │
    ▼ トークナイザ
[1, 234, 56, 789, 12, 2, 1]  (トークン ID 列)
    │
    ▼ Embedding + Position
[[0.02, -0.1, ...], [0.5, 0.3, ...], ...]  (384 次元ベクトルの列)
    │
    ▼ Transformer Block ×6 (Attention + FFN)
[[0.8, -0.3, ...], [0.1, 0.7, ...], ...]  (文脈を反映した 384 次元ベクトルの列)
    │
    ▼ LM Head
[0.01, 0.001, ..., 0.15, ..., 0.002]  (4096 次元: 各トークンの確率)
    │
    ▼ temperature + top_k + サンプリング
トークン ID: 345
    │
    ▼ デコード
"hello"
    │
    （これを繰り返して文を生成）
```

---

## ファイル別の役割まとめ

| ファイル | 役割 | 行数 | 一言 |
|---|---|---|---|
| `config.py` | ハイパーパラメータ定義 | 37 | モデルと学習の設定値 |
| `generate_data.py` | 学習データ生成 | ~600 | テンプレート × ランダムで 60K 件 |
| `prepare_data.py` | データ準備 + トークナイザ学習 | 74 | BPE トークナイザをゼロから学習 |
| `dataset.py` | PyTorch Dataset/DataLoader | 52 | JSONL → トークン列 → バッチ化 |
| `model.py` | Transformer モデル本体 | 130 | Embedding → Block×6 → LM Head |
| `train.py` | 学習ループ | 161 | AdamW + cosine LR + AMP |
| `inference.py` | 推論・チャット | 127 | autoregressive 生成 + ChatML |
| `eval_cases.py` | 評価用テストケース | 126 | 手書きの品質チェック用 |
| `__main__.py` | CLI エントリポイント | 77 | prepare / train / chat / download |

---

## あなたの知識との対応表

| あなたが知っていること | GuppyLM での対応 |
|---|---|
| llama-server の `--ctx-size` | `max_seq_len = 128` |
| llama.cpp の `--temp`, `--top-k` | `generate()` の `temperature`, `top_k` |
| ChatML テンプレート | `_format_prompt()` と `generate_data.py` の出力形式 |
| GGUF のトークナイザ | `data/tokenizer.json`（BPE、ただし語彙 4,096） |
| KV cache | 未実装（128 トークンなので毎回再計算で十分） |
| LoRA / QLoRA | 不使用（ゼロから学習、パラメータが少ないので全パラメータ更新） |
| llama.cpp の量子化 (Q4_K_M 等) | `tools/export_onnx.py` で uint8 量子化（ブラウザ用） |

---

## なぜ「これだけ」で動くのか

1. **タスクが極めて限定的**: 魚のキャラクターとして短文で返答するだけ。汎用的な知識は不要。
2. **データとモデルのバランス**: 60K サンプル × 128 トークン × 10,000 ステップ。小さなモデルに小さなデータを十分な回数見せている。
3. **テンプレートの多様性**: ランダム組み合わせで同じ質問にも多様な返答パターンがあり、丸暗記ではなくパターン学習が起きる。
4. **Transformer の表現力**: Self-Attention は「どの単語がどの単語に関連するか」を柔軟に学習でき、たった 6 層でも短い会話なら十分に機能する。
