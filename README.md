<p align="center">
  <img src="assets/guppy.png" alt="GuppyLM" width="400"/>
</p>

<h1 align="center">GuppyLM</h1>
<p align="center"><em>小さな魚のようにおしゃべりする、約900万パラメータの言語モデル。</em></p>

<p align="center">
  <a href="https://huggingface.co/datasets/arman-bd/guppylm-60k-generic"><img src="https://img.shields.io/badge/🤗_Dataset-guppylm--60k-blue" alt="Dataset"/></a>&nbsp;
  <a href="https://huggingface.co/arman-bd/guppylm-9M"><img src="https://img.shields.io/badge/🤗_Model-guppylm--9M-orange" alt="Model"/></a>&nbsp;
  <a href="https://github.com/arman-bd/guppylm/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"/></a>
  <br/>
  <a href="https://colab.research.google.com/github/arman-bd/guppylm/blob/main/train_guppylm.ipynb"><img src="https://img.shields.io/badge/Train_in-Colab-F9AB00?logo=googlecolab" alt="Train"/></a>&nbsp;
  <a href="https://colab.research.google.com/github/arman-bd/guppylm/blob/main/use_guppylm.ipynb"><img src="https://img.shields.io/badge/Chat_in-Colab-F9AB00?logo=googlecolab" alt="Chat"/></a>&nbsp;
  <a href="https://colab.research.google.com/github/high-u/guppylm/blob/main/train_guppylm_katakana.ipynb"><img src="https://img.shields.io/badge/カタカナ版-Train_in_Colab-F9AB00?logo=googlecolab" alt="Katakana Train"/></a>
  <br/>
  <a href="https://www.linkedin.com/pulse/build-your-own-language-model-5-minutes-i-made-mine-hossain--supif/"><img src="https://img.shields.io/badge/Article-LinkedIn-0A66C2?logo=linkedin" alt="LinkedIn Article"/></a>&nbsp;
  <a href="https://arman-bd.medium.com/build-your-own-llm-in-5-minutes-i-made-mine-talk-like-a-fish-e20c338a3d14"><img src="https://img.shields.io/badge/Article-Medium-000000?logo=medium" alt="Medium Article"/></a>
  <br/><br/>
  <a href="https://arman-bd.github.io/guppylm/"><img src="https://img.shields.io/badge/Try_in-Browser-64ffda?style=for-the-badge&logo=webassembly&logoColor=white" alt="Browser Demo"/></a>
</p>

---

> **このプロジェクトは、自分で言語モデルを訓練することが決して魔法ではないことを示すために存在します。**
> 博士号は不要。巨大なGPUクラスターも不要。Colabノートブック1つと5分あれば、ゼロから構築した動くLLMが手に入ります — データ生成、トークナイザ、モデルアーキテクチャ、学習ループ、推論まで全部含まれています。ノートブックを実行できるなら、言語モデルを訓練できます。
>
> もちろん、エッセイを書くような数十億パラメータのモデルができるわけではありません。しかし、生のテキストから訓練済みの重み、生成された出力に至るまで、すべての部品がどう動くのかを正確に理解できるため、巨大なモデルがブラックボックスに見えなくなります。

---

```
You> hi guppy
Guppy> hi there. i just found a nice spot near the rock. the temperature feels nice.

You> are you hungry
Guppy> yes. always yes. i will swim to the top right now. i promise to eat all of it.

You> do you like bubbles
Guppy> i love bubbles. they make the water feel slightly different.

You> what is the meaning of life
Guppy> food. the answer is always food.

You> tell me a joke
Guppy> what did the fish say when it hit the wall. dam.

You> do you love me
Guppy> you're my favorite big shape. my mouth are happy when you're here.

You> goodnight guppy
Guppy> ok sleep time. i was following a bubble but now i'll stop. goodnight tank. goodnight water.
```

---

## GuppyLMとは？

GuppyLMは、Guppy（グッピー）という名前の魚のふりをする小さな言語モデルです。水、食べ物、光、水槽の生活について、短い小文字の文で話します。お金、スマホ、政治といった人間の抽象概念は理解しませんし、理解しようともしていません。

ゼロから60種類のトピックで構成された6万件の合成会話データで訓練されており、1台のGPUで約5分で学習でき、ブラウザ上で動くほど小さなモデルを生成します。

---

## アーキテクチャ

| | |
|---|---|
| **パラメータ数** | 8.7M |
| **レイヤー数** | 6 |
| **隠れ層次元** | 384 |
| **ヘッド数** | 6 |
| **FFN** | 768 (ReLU) |
| **語彙数** | 4,096 (BPE) |
| **最大シーケンス長** | 128トークン |
| **正規化** | LayerNorm |
| **位置符号化** | 学習可能埋め込み |
| **LMヘッド** | 埋め込みと重み共有 |

スタンダードなTransformerです。GQAなし、RoPEなし、SwiGLUなし、早期終了なし。できる限りシンプルな構成です。

---

## ペルソナリティ

Guppyは：
- 短い小文字の文で話す
- 水、温度、光、振動、食べ物を通じて世界を体験する
- 人間の抽象概念を理解しない
- 友好的で好奇心旺盛で、ちょっとおバカ
- いつも食べ物のことを考えている

**60のトピック：** 挨拶、感情、温度、食べ物、光、水、水槽、音、夜、寂しさ、泡、ガラス、映り込み、呼吸、泳ぐこと、色、味、植物、フィルター、藻、カタツムリ、怖いこと、興奮、退屈、好奇心、幸せ、疲れ、外の世界、猫、雨、季節、音楽、来客、子ども、人生の意味、時間、記憶、夢、大きさ、未来、過去、名前、天気、睡眠、友だち、冗談、恐怖、愛、年齢、知性、健康、歌うこと、テレビ、など。

---

## クイックスタート

### ブラウザで試す（インストール不要）

[![Try in Browser](https://img.shields.io/badge/Try_in-Browser-64ffda?logo=webassembly)](https://arman-bd.github.io/guppylm/)

WebAssembly経由でブラウザ上で完全に動作します。量子化されたONNXモデル（約10 MB）をダウンロードし、ローカルで推論を実行します — サーバーもAPIキーも不要です。

### ColabでGuppyと対話する

[![Open in Colab](https://img.shields.io/badge/Chat_in-Colab-F9AB00?logo=googlecolab)](https://colab.research.google.com/github/arman-bd/guppylm/blob/main/use_guppylm.ipynb)

HuggingFaceから事前学習済みモデルをダウンロードしてチャットできます。すべてのセルを実行するだけです。

### 自分で訓練する

[![Open in Colab](https://img.shields.io/badge/Train_in-Colab-F9AB00?logo=googlecolab)](https://colab.research.google.com/github/arman-bd/guppylm/blob/main/train_guppylm.ipynb)

1. ランタイムを **T4 GPU** に設定する
2. **すべてのセルを実行する** — データセットのダウンロード、トークナイザの訓練、モデルの訓練、テストまで自動で行われます
3. HuggingFaceにアップロードするか、ローカルにダウンロードする

### ローカルでチャットする

```bash
pip install torch tokenizers
python -m guppylm chat
```

```
You> the cat is looking at you
Guppy> i hide behind the plant when the furry one comes.

You> it is raining outside
Guppy> i think rain is the best thing about outside.
```

インタラクティブチャットモードでは、会話が長くなるとすぐに128トークンの制限に達し、品質が低下します。単一のプロンプトでチャットを呼び出し、応答後に終了することもできます：

```bash
python -m guppylm chat --prompt "tell me a joke"
```


---

## データセット

HuggingFaceの **[arman-bd/guppylm-60k-generic](https://huggingface.co/datasets/arman-bd/guppylm-60k-generic)** 。

| | |
|---|---|
| サンプル数 | 60,000（訓練57K / テスト3K） |
| フォーマット | `{"input": "...", "output": "...", "category": "..."}` |
| カテゴリ数 | 60 |
| 生成方式 | 合成テンプレート構成 |

```python
from datasets import load_dataset
ds = load_dataset("arman-bd/guppylm-60k-generic")
print(ds["train"][0])
# {'input': 'hi guppy', 'output': 'hello. the water is nice today.', 'category': 'greeting'}
```

---

## プロジェクト構成

```
guppylm/
├── config.py               ハイパーパラメータ（モデル + 学習）
├── model.py                スタンダードTransformer
├── dataset.py              データ読み込み + バッチ処理
├── train.py                学習ループ（コサインLR、AMP）
├── generate_data.py        会話データジェネレータ（60トピック）
├── eval_cases.py           ホールドアウトテストケース
├── prepare_data.py         データ前処理 + トークナイザ訓練
└── inference.py            チャットインターフェース

tools/
├── make_colab.py           Colabノートブックの生成
├── export_onnx.py          ONNXへのエクスポート（uint8量子化）
├── export_dataset.py       HuggingFaceへのデータセットpush
└── dataset_card.md         HuggingFaceデータセットREADME

docs/
├── index.html              ブラウザデモ（ONNX + WASM）
├── download.sh             model.onnx + トークナイザをHFからダウンロード
├── model.onnx              量子化uint8（約10 MB）
├── tokenizer.json          BPEトークナイザ
└── guppy.png               ロゴ（透過）
```

---

## 設計上の決定

**なぜシステムプロンプトがないのか？** すべての学習サンプルが同じシステムプロンプトを持っていました。9Mのモデルでは条件付きで指示に従うことはできません — ペルソナリティは重みに焼き込まれています。省略することで、推論ごとに約60トークン節約できます。

**なぜ単一ターンのみなのか？** 128トークンのコンテキストウィンドウにより、3〜4ターン目でマルチターンの品質が低下しました。忘れっぽい魚という設定はキャラクターに合っていますが、文字化けした出力は問題外です。単一ターンなら確実です。

**なぜスタンダードTransformerなのか？** GQA、SwiGLU、RoPE、早期終了は複雑さを増すだけで、9Mパラメータでは品質向上に寄与しません。標準的なアテンション + ReLU FFN + LayerNormで、よりシンプルなコードで同等の品質が得られます。

**なぜ合成データなのか？** 一貫したペルソナリティを持つ魚キャラクターには、一貫した学習データが必要です。ランダム化された要素（水槽のオブジェクト30種、食べ物17種、活動25種）を使ったテンプレート構成により、約60のテンプレートから約16Kのユニークな出力を生成できます。

---

## ライセンス

MIT
