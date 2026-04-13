<p align="center">
  <img src="assets/guppy.png" alt="GuppyLM" width="400"/>
</p>

<h1 align="center">GuppyLM（カタカナ版）</h1>
<p align="center"><em>小さな魚のようにおしゃべりする、約900万パラメータの言語モデル。</em></p>

<p align="center">
  <a href="https://colab.research.google.com/github/high-u/guppylm/blob/hepburn/train_guppylm_katakana.ipynb"><img src="https://img.shields.io/badge/カタカナ版-Train_in_Colab-F9AB00?logo=googlecolab" alt="Katakana Train"/></a>
</p>

---

> **このプロジェクトは、自分で言語モデルを訓練することが決して魔法ではないことを示すために存在します。**
> 博士号は必要ありません。巨大なGPUクラスターも必要ありません。Colabノートブック1つと5分あれば、ゼロから構築した動くLLMが手に入ります — データ生成、トークナイザ、モデルアーキテクチャ、学習ループ、推論まで全部含まれています。ノートブックを実行できるなら、言語モデルを訓練できます。
>
> もちろん、エッセイを書くような数十億パラメータのモデルができるわけではありません。しかし、生のテキストから訓練済みの重み、生成された出力に至るまで、すべての部品がどう動くのかを正確に理解できるため、巨大なモデルがブラックボックスに見えなくなります。
>
> <p align="right">— <a href="https://github.com/arman-bd/guppylm">arman-bd/guppylm</a> より</p>

---

```
You> コンニチハ グッピー
Guppy> オハヨウ。ミズ ガ キモチイイ ネ。ゲンキ？

You> エサ アゲル ネ
Guppy> ウレシイ。オナカ スイテタ ヨ。フレーク オイシイ。

You> サムイ ケド ダイジョウブ?
Guppy> ミズ ガ ツメタイ ヨ。オンド カワッタ ネ。

You> イキル イミ ハ？
Guppy> エサ。タベル コト。ソレ ガ イッパイ。

You> オヤスミ グッピー
Guppy> オヤスミ。ヨル ハ シズカ ダ ネ。マタ アシタ ネ。
```

---

## GuppyLMカタカナ版とは？

[arman-bd/guppylm](https://github.com/arman-bd/guppylm) のフォークで、グッピーが**カタカナ**で話すバージョンです。水、食べ物、光、水槽の生活についてカタカナの文で話します。お金、スマホ、政治といった人間の抽象概念は理解しませんし、理解しようともしていません。

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
- カタカナで話す
- 水、温度、光、振動、食べ物を通じて世界を体験する
- 人間の抽象概念は理解できない
- 友好的で好奇心旺盛で、ちょっとおバカ
- いつも食べ物のことを考えている

**60のトピック：** 挨拶、別れ、感情、幸福、興奮、恐怖、退屈、疲れ、好奇心、寂しさ、食べ物、味、水、フィルター、藻、泡、暑さ、寒さ、天気、雨、季節、光、夜、睡眠、時間、音、ガラス、音楽、歌うこと、水槽、植物、ガラス越し、映り込み、外の世界、自分のこと、名前、大きさ、年齢、人生の意味、夢、記憶、未来、過去、猫、カタツムリ、友だち、来客、子ども、愛、呼吸、泳ぐこと、色、うんち、お医者さん、混乱、知性、冗談、テレビ、恐怖、目標、雑談。

---

## クイックスタート

### ブラウザで試す（インストール不要）

[![ブラウザで試す](https://img.shields.io/badge/ブラウザで試す-64ffda?logo=webassembly)](https://high-u.github.io/guppylm/)

WebAssembly経由でブラウザ上で完全に動作します。量子化されたONNXモデル（約9 MB）をダウンロードし、ローカルで推論を実行します — サーバーもAPIキーも不要です。

### 自分で訓練する

[![Open in Colab](https://img.shields.io/badge/カタカナ版-Train_in_Colab-F9AB00?logo=googlecolab)](https://colab.research.google.com/github/high-u/guppylm/blob/hepburn/train_guppylm_katakana.ipynb)

1. ランタイムを **T4 GPU** に設定する
2. **すべてのセルを実行する** — データセットの生成、トークナイザの訓練、モデルの訓練、テストまで自動で行われます
3. チェックポイントは `checkpoints/` に保存されます

### ローカルでチャットする

```bash
pip install -r requirements.txt
python -m guppylm prepare    # データ生成 + トークナイザ訓練
python -m guppylm train      # モデルの訓練
python -m guppylm chat       # チャット開始
```

```
You> ナニ カンガエテル?
Guppy> エサ ノ コト。イシ ノ アタリ ヲ オヨイデイル。

You> ネコ ガ ミテル ヨ
Guppy> ネコ コワイ。イシ ノ ウラ ニ カクレル。
```

safetensorsフォーマットを使う場合：

```bash
python -m guppylm chat --safetensors
```

単一のプロンプトで呼び出すこともできます：

```bash
python -m guppylm chat --prompt "ジョーダン オシエテ"
```

### ONNXエクスポート

```bash
pip install onnx onnxruntime
python -m guppylm export              # uint8量子化（約9 MB）
python -m guppylm export --no-quantize  # float32（約35 MB）
```

---

## データ生成

カタカナ版では、[単語辞書 + 文法パターン] の組み合わせで応答を生成します。

| | |
|---|---|
| サンプル数 | 60,000（訓練57K / テスト3K） |
| フォーマット | `{"input": "...", "output": "...", "category": "..."}` |
| カテゴリ数 | 60 |
| 生成方式 | 単語辞書（品詞タグ付き）+ 文法パターン（P1-P8）の組み合わせ |

**文法パターン:**

| パターン | 構造 | 例 |
|---|---|---|
| P1 | 形容詞 | オイシイ |
| P2 | N ガ V | エサ ガ アル |
| P3 | Adj N | オイシイ フレーク |
| P4 | N ヲ V（他動詞） | エサ ヲ タベル |
| P5 | V | オヨイデイル |
| P6 | N デ V | スイソウ デ ネル |
| P7 | N ニ V | イシ ニ カクレル |
| P8 | N ハ V | エサ ハ スキ |

---

## プロジェクト構成

```
guppylm/
├── __init__.py              GuppyConfig, TrainConfig, GuppyLM のエクスポート
├── __main__.py              CLI エントリポイント（train, prepare, chat, export）
├── config.py                ハイパーパラメータ（モデル + 学習）
├── model.py                 スタンダードTransformer
├── dataset.py               データ読み込み + バッチ処理
├── train.py                 学習ループ（コサインLR、AMP、safetensors保存）
├── inference.py             チャットインターフェース（.pt / safetensors対応）
├── generate_data.py         カタカナ会話データジェネレータ（単語辞書 + 文法パターン）
├── eval_cases.py            ホールドアウトテストケース（16件）
├── prepare_data.py          データ前処理 + トークナイザ訓練
└── export_onnx.py           ONNXへのエクスポート（uint8量子化対応）

public/
├── index.html               ブラウザデモ（ONNX + WASM、アニメーション付きUI）
├── model.onnx               量子化uint8（約9 MB）
├── tokenizer.json           BPEトークナイザ
├── guppy.svg                グッピーキャラクター
├── seaweed1.svg             海藻デコレーション
└── seaweed2.svg             海藻デコレーション

data/                        学習データ（prepare で生成）
checkpoints/                 モデルチェックポイント（.pt / .safetensors）
```

---

## 設計上の決定

**なぜシステムプロンプトがないのか？** すべての学習サンプルが同じペルソナを持っています。9Mのモデルでは条件付きで指示に従うことはできません — ペルソナリティは重みに焼き込まれています。省略することで、推論ごとに約60トークン節約できます。

**なぜ単一ターンのみなのか？** 128トークンのコンテキストウィンドウにより、3〜4ターン目でマルチターンの品質が低下しました。忘れっぽい魚という設定はキャラクターに合っていますが、文字化けした出力は問題外です。単一ターンなら確実です。

**なぜスタンダードTransformerなのか？** GQA、SwiGLU、RoPE、早期終了は複雑さを増すだけで、9Mパラメータでは品質向上に寄与しません。標準的なアテンション + ReLU FFN + LayerNormで、よりシンプルなコードで同等の品質が得られます。

**なぜ単語辞書 + 文法パターンなのか？** 一貫したペルソナリティを持つカタカナ魚キャラクターには、一貫した学習データが必要です。品詞タグ付きの単語辞書（約450語）と8種の文法パターンを組み合わせることで、60トピックから6万件のユニークな会話を生成できます。

---

## フォーク元との違い

このリポジトリは [arman-bd/guppylm](https://github.com/arman-bd/guppylm) のフォークです。主な変更点：

- **カタカナ対応** — データ生成、ペルソナリティ、UIをすべてカタカナ化
- **パッケージ統合** — `tools/` を `guppylm/` パッケージに統合し、CLIコマンドを追加
- **safetensors対応** — 学習・推論の両方でsafetensorsフォーマットをサポート
- **ブラウザデモの刷新** — アニメーション付きUI、カタカナ入力バリデーション、SVGアセット
- **ONNX量子化** — uint8量子化による約4倍のサイズ削減（35MB → 9MB）

---

## ライセンス

MIT
