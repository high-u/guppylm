# カタカナ データ生成仕様

## 目的

GuppyLM の学習データをカタカナで生成する。
成功基準: **ユーザーのトピックからズレない回答をモデルが返すこと。**

## 方針

- 英語版の `generate_data.py` のテンプレート手書き方式ではなく、**タグ付き単語辞書 + 文法パターンによる組み立て**で文を生成する
- ユーザー側: トピックごとに具体的な発話パターンを手書きで用意する（英語版と同じ方式）
- アシスタント側: 単語辞書から文法パターンで組み立てる
- カタカナ表記、助詞はカタカナそのまま (「ハ」「ヘ」「ヲ」)
- 単語間はスペースで区切る (例: `エサ ハ オイシイ`)
- 助詞も独立させる (例: `イシ ノ ウシロ デ`)

---

## 1. 単語辞書

### 1.1 データ構造

単語辞書は Python リストで定義する。各エントリは辞書型。

```python
WORD_DICT = [
    {"word": "エサ", "pos": "noun", "topics": ["food", "taste"]},
    {"word": "タベル", "pos": "verb", "topics": ["food", "taste"]},
    ...
]
```

### 1.2 品詞 (pos)

| pos | 説明 | 文法パターンでの使い方 |
|---|---|---|
| noun | 名詞 | P1〜P4, P6 のスロットに入る |
| verb | 動詞 (活用形ごとに別エントリ) | P2, P4, P5, P6 のスロットに入る |
| adj | 形容詞 | P1, P3 のスロットに入る |
| reaction | 応答・感嘆 (文1専用) | 文1 にのみ使用。文法パターンには入らない |

### 1.3 動詞の活用形

動詞は活用形ごとに別エントリで登録する。活用ロジックは実装しない。

登録する活用形:
- 辞書形: タベル, オヨグ
- タイ形: タベタイ, オヨギタイ
- テイル形: タベテイル, オヨイデイル
- タ形: タベタ, オイダ

**全ての活用形は同じ pos=verb として登録する。** 文法パターン側で「どの活用形を使うか」は制約しない。どの verb エントリも P2, P4, P5, P6 に入れてよい。

理由: パターンとの組み合わせで多少不自然になるケースがあっても、トピック一致の学習には影響しない。自然さより実装のシンプルさを優先する。

### 1.4 単語とトピックの関係

単語とトピックは **N:M の関係**。

- 1つの単語が複数のトピックに属する (例: 「ミズ」→ water, filter, temp_hot, temp_cold, bubbles)
- 1つのトピックに複数の単語が属する (例: food → エサ, フレーク, タベル, オイシイ...)

生成時、指定されたトピックに属する単語のみをプールから選択する。
同じトピックで生成しても、ランダム選択 × パターン選択で毎回異なる応答になる。
異なるトピック間で語彙が重なるため似た応答が出ることがあるが、確率的であり問題としない。

### 1.5 トピック一覧とグループ

60 トピックを 13 グループに分類する。グループはトピック間で語彙を共有するための整理単位であり、生成ロジックには影響しない。生成は常にトピック単位で行う。

| グループ | トピック |
|---|---|
| 挨拶・別れ | greeting, bye |
| 感情・状態 | feeling, happy, excited, scared, bored, tired, curious, lonely |
| 食 | food, taste |
| 水・環境 | water, filter, algae, bubbles |
| 温度 | temp_hot, temp_cold, weather, rain, seasons |
| 光・時間 | light, night, sleep, time |
| 音・振動 | noise, glass_tap, music, singing |
| 水槽・空間 | tank, plants, glass, reflection, outside |
| 自己・存在 | about, name, size, age, meaning, dreams, memory, future, past |
| 他者 | cat, snail, friends, visitors, children, love |
| 身体 | breathing, swimming, colors, poop, doctor |
| 不明 | confused, smart, joke, tv, fear |
| その他 | misc |

### 1.6 全単語辞書

以下が完全な単語辞書である。実装時はこの定義をそのまま使用すること。

#### 挨拶・別れグループ

```python
# noun
{"word": "アサ",       "pos": "noun", "topics": ["greeting"]},
{"word": "ヨル",       "pos": "noun", "topics": ["bye"]},
{"word": "ヒル",       "pos": "noun", "topics": ["greeting"]},
{"word": "キョウ",     "pos": "noun", "topics": ["greeting", "bye"]},

# verb
{"word": "キタ",       "pos": "verb", "topics": ["greeting"]},
{"word": "マッテタ",   "pos": "verb", "topics": ["greeting"]},
{"word": "マタ クル",  "pos": "verb", "topics": ["bye"]},
{"word": "マッテル",   "pos": "verb", "topics": ["bye"]},
{"word": "イル",       "pos": "verb", "topics": ["greeting", "bye"]},

# adj
{"word": "ウレシイ",   "pos": "adj", "topics": ["greeting"]},
{"word": "サミシイ",   "pos": "adj", "topics": ["bye"]},
{"word": "ゲンキ",     "pos": "adj", "topics": ["greeting"]},

# reaction
{"word": "ヤッホー",   "pos": "reaction", "topics": ["greeting"]},
{"word": "オハヨウ",   "pos": "reaction", "topics": ["greeting"]},
{"word": "コンニチハ", "pos": "reaction", "topics": ["greeting"]},
{"word": "バイバイ",   "pos": "reaction", "topics": ["bye"]},
{"word": "オヤスミ",   "pos": "reaction", "topics": ["bye"]},
{"word": "マタネ",     "pos": "reaction", "topics": ["bye"]},
```

#### 感情・状態グループ

```python
# noun
{"word": "キブン",     "pos": "noun", "topics": ["feeling", "happy", "bored", "tired"]},
{"word": "キモチ",     "pos": "noun", "topics": ["feeling", "happy", "scared", "lonely"]},
{"word": "カラダ",     "pos": "noun", "topics": ["tired", "feeling"]},
{"word": "ココロ",     "pos": "noun", "topics": ["lonely", "happy", "scared"]},
{"word": "ヒマ",       "pos": "noun", "topics": ["bored"]},
{"word": "ナミダ",     "pos": "noun", "topics": ["lonely"]},

# verb
{"word": "ワクワク シテイル",  "pos": "verb", "topics": ["excited", "curious"]},
{"word": "ドキドキ シテイル",  "pos": "verb", "topics": ["scared", "excited"]},
{"word": "ノンビリ スル",      "pos": "verb", "topics": ["bored", "tired"]},
{"word": "カンガエル",         "pos": "verb", "topics": ["curious", "feeling"]},
{"word": "サガシテイル",       "pos": "verb", "topics": ["curious"]},
{"word": "ヤスム",             "pos": "verb", "topics": ["tired"]},
{"word": "ヤスミタイ",         "pos": "verb", "topics": ["tired"]},
{"word": "ガンバル",           "pos": "verb", "topics": ["feeling", "excited"]},
{"word": "フルエテイル",       "pos": "verb", "topics": ["scared"]},
{"word": "ナイテイル",         "pos": "verb", "topics": ["lonely"]},

# adj
{"word": "タノシイ",   "pos": "adj", "topics": ["happy", "excited"]},
{"word": "コワイ",     "pos": "adj", "topics": ["scared"]},
{"word": "ツマラナイ", "pos": "adj", "topics": ["bored"]},
{"word": "ツカレタ",   "pos": "adj", "topics": ["tired"]},
{"word": "キニナル",   "pos": "adj", "topics": ["curious"]},
{"word": "サミシイ",   "pos": "adj", "topics": ["lonely", "bye"]},
{"word": "シアワセ",   "pos": "adj", "topics": ["happy"]},
{"word": "フアン",     "pos": "adj", "topics": ["scared", "lonely"]},
{"word": "ネムイ",     "pos": "adj", "topics": ["tired"]},
{"word": "オモシロイ", "pos": "adj", "topics": ["curious", "excited"]},
{"word": "イイ",       "pos": "adj", "topics": ["feeling", "happy"]},

# reaction
{"word": "ウン",       "pos": "reaction", "topics": ["feeling"]},
{"word": "ヤッタ",     "pos": "reaction", "topics": ["happy", "excited"]},
{"word": "キャー",     "pos": "reaction", "topics": ["scared"]},
{"word": "ヒマダナ",   "pos": "reaction", "topics": ["bored"]},
{"word": "ツカレタ",   "pos": "reaction", "topics": ["tired"]},
{"word": "エー",       "pos": "reaction", "topics": ["curious"]},
{"word": "ショボン",   "pos": "reaction", "topics": ["lonely"]},
{"word": "マアマア",   "pos": "reaction", "topics": ["feeling"]},
{"word": "ウレシイ",   "pos": "reaction", "topics": ["happy", "excited"]},
```

#### 食グループ

```python
# noun
{"word": "エサ",       "pos": "noun", "topics": ["food", "taste"]},
{"word": "フレーク",   "pos": "noun", "topics": ["food"]},
{"word": "ペレット",   "pos": "noun", "topics": ["food"]},
{"word": "オナカ",     "pos": "noun", "topics": ["food"]},
{"word": "クチ",       "pos": "noun", "topics": ["food", "taste"]},
{"word": "アジ",       "pos": "noun", "topics": ["taste"]},
{"word": "ゴハン",     "pos": "noun", "topics": ["food"]},

# verb
{"word": "タベル",     "pos": "verb", "topics": ["food", "taste"]},
{"word": "タベタイ",   "pos": "verb", "topics": ["food"]},
{"word": "タベテイル", "pos": "verb", "topics": ["food", "taste"]},
{"word": "タベタ",     "pos": "verb", "topics": ["food", "taste"]},
{"word": "カム",       "pos": "verb", "topics": ["food", "taste"]},
{"word": "ノム",       "pos": "verb", "topics": ["taste"]},
{"word": "マッテイル", "pos": "verb", "topics": ["food"]},

# adj
{"word": "オイシイ",   "pos": "adj", "topics": ["food", "taste"]},
{"word": "スキ",       "pos": "adj", "topics": ["food", "taste"]},
{"word": "オナカスイタ", "pos": "adj", "topics": ["food"]},

# reaction
{"word": "ウレシイ",   "pos": "reaction", "topics": ["food"]},
{"word": "ヤッタ",     "pos": "reaction", "topics": ["food"]},
{"word": "オナカスイタ", "pos": "reaction", "topics": ["food"]},
```

#### 水・環境グループ

```python
# noun
{"word": "ミズ",       "pos": "noun", "topics": ["water", "filter", "bubbles", "temp_hot", "temp_cold", "algae"]},
{"word": "アワ",       "pos": "noun", "topics": ["bubbles", "water"]},
{"word": "フィルター", "pos": "noun", "topics": ["filter", "water"]},
{"word": "スイソウ",   "pos": "noun", "topics": ["water", "filter", "algae"]},
{"word": "コケ",       "pos": "noun", "topics": ["algae"]},
{"word": "ナガレ",     "pos": "noun", "topics": ["filter", "water", "bubbles"]},

# verb
{"word": "ナガレテイル", "pos": "verb", "topics": ["water", "filter", "bubbles"]},
{"word": "キレイニ ナッタ", "pos": "verb", "topics": ["water", "filter"]},
{"word": "ニゴッテイル",   "pos": "verb", "topics": ["water", "algae"]},
{"word": "デテイル",       "pos": "verb", "topics": ["bubbles"]},
{"word": "ウゴイテイル",   "pos": "verb", "topics": ["filter"]},
{"word": "フエテイル",     "pos": "verb", "topics": ["algae", "bubbles"]},
{"word": "イキ ガ シヤスイ", "pos": "verb", "topics": ["water", "filter"]},

# adj
{"word": "キレイ",     "pos": "adj", "topics": ["water", "filter"]},
{"word": "ニゴッテル", "pos": "adj", "topics": ["water", "algae"]},
{"word": "スズシイ",   "pos": "adj", "topics": ["water"]},
{"word": "キモチイイ", "pos": "adj", "topics": ["water", "bubbles"]},
{"word": "シズカ",     "pos": "adj", "topics": ["filter"]},

# reaction
{"word": "キモチイイ", "pos": "reaction", "topics": ["water", "bubbles"]},
{"word": "サッパリ",   "pos": "reaction", "topics": ["water", "filter"]},
{"word": "アリガトウ", "pos": "reaction", "topics": ["water", "filter"]},
{"word": "ウワー",     "pos": "reaction", "topics": ["bubbles"]},
{"word": "クサイ",     "pos": "reaction", "topics": ["algae"]},
```

#### 温度グループ

```python
# noun
{"word": "オンド",     "pos": "noun", "topics": ["temp_hot", "temp_cold"]},
{"word": "ナツ",       "pos": "noun", "topics": ["temp_hot", "seasons"]},
{"word": "フユ",       "pos": "noun", "topics": ["temp_cold", "seasons"]},
{"word": "ハル",       "pos": "noun", "topics": ["seasons"]},
{"word": "アキ",       "pos": "noun", "topics": ["seasons"]},
{"word": "アメ",       "pos": "noun", "topics": ["rain", "weather"]},
{"word": "ソラ",       "pos": "noun", "topics": ["weather", "rain"]},
{"word": "カゼ",       "pos": "noun", "topics": ["weather", "temp_cold"]},
{"word": "ヒ",         "pos": "noun", "topics": ["temp_hot", "weather"]},
{"word": "ユキ",       "pos": "noun", "topics": ["temp_cold", "weather", "seasons"]},

# verb
{"word": "アツク ナッタ",    "pos": "verb", "topics": ["temp_hot"]},
{"word": "ツメタク ナッタ",  "pos": "verb", "topics": ["temp_cold"]},
{"word": "フッテイル",       "pos": "verb", "topics": ["rain"]},
{"word": "カワッタ",         "pos": "verb", "topics": ["seasons", "weather"]},
{"word": "アガッタ",         "pos": "verb", "topics": ["temp_hot"]},
{"word": "サガッタ",         "pos": "verb", "topics": ["temp_cold"]},
{"word": "フルエテイル",     "pos": "verb", "topics": ["temp_cold"]},

# adj
{"word": "アツイ",     "pos": "adj", "topics": ["temp_hot"]},
{"word": "ツメタイ",   "pos": "adj", "topics": ["temp_cold"]},
{"word": "アタタカイ", "pos": "adj", "topics": ["temp_hot", "seasons"]},
{"word": "サムイ",     "pos": "adj", "topics": ["temp_cold", "weather"]},
{"word": "スズシイ",   "pos": "adj", "topics": ["temp_cold", "seasons"]},
{"word": "ムシアツイ", "pos": "adj", "topics": ["temp_hot", "weather", "rain"]},

# reaction
{"word": "アツイ",     "pos": "reaction", "topics": ["temp_hot"]},
{"word": "サムイ",     "pos": "reaction", "topics": ["temp_cold"]},
{"word": "イヤダ",     "pos": "reaction", "topics": ["temp_hot", "temp_cold"]},
{"word": "ツライ",     "pos": "reaction", "topics": ["temp_hot", "temp_cold"]},
{"word": "キモチイイ", "pos": "reaction", "topics": ["seasons", "weather"]},
{"word": "アメダ",     "pos": "reaction", "topics": ["rain"]},
```

#### 光・時間グループ

```python
# noun
{"word": "ヒカリ",     "pos": "noun", "topics": ["light", "night"]},
{"word": "クラヤミ",   "pos": "noun", "topics": ["night"]},
{"word": "アサ",       "pos": "noun", "topics": ["light", "time"]},
{"word": "ヨル",       "pos": "noun", "topics": ["night", "sleep", "time"]},
{"word": "ジカン",     "pos": "noun", "topics": ["time"]},
{"word": "ネムリ",     "pos": "noun", "topics": ["sleep"]},

# verb
{"word": "ツイタ",     "pos": "verb", "topics": ["light"]},
{"word": "キエタ",     "pos": "verb", "topics": ["light", "night"]},
{"word": "ネル",       "pos": "verb", "topics": ["sleep", "night"]},
{"word": "ネテイル",   "pos": "verb", "topics": ["sleep"]},
{"word": "ネムイ",     "pos": "verb", "topics": ["sleep", "tired"]},
{"word": "オキタ",     "pos": "verb", "topics": ["sleep", "light"]},
{"word": "スギル",     "pos": "verb", "topics": ["time"]},
{"word": "マブシイ",   "pos": "verb", "topics": ["light"]},

# adj
{"word": "アカルイ",   "pos": "adj", "topics": ["light"]},
{"word": "クライ",     "pos": "adj", "topics": ["night"]},
{"word": "マブシイ",   "pos": "adj", "topics": ["light"]},
{"word": "シズカ",     "pos": "adj", "topics": ["night", "sleep"]},
{"word": "ハヤイ",     "pos": "adj", "topics": ["time"]},
{"word": "オソイ",     "pos": "adj", "topics": ["time"]},
{"word": "ネムイ",     "pos": "adj", "topics": ["sleep", "tired"]},

# reaction
{"word": "マブシイ",   "pos": "reaction", "topics": ["light"]},
{"word": "クライ",     "pos": "reaction", "topics": ["night"]},
{"word": "オヤスミ",   "pos": "reaction", "topics": ["sleep", "night"]},
{"word": "ネムイ",     "pos": "reaction", "topics": ["sleep"]},
{"word": "モウ コンナ ジカン", "pos": "reaction", "topics": ["time"]},
```

#### 音・振動グループ

```python
# noun
{"word": "オト",       "pos": "noun", "topics": ["noise", "glass_tap", "music", "singing"]},
{"word": "シンドウ",   "pos": "noun", "topics": ["noise", "glass_tap"]},
{"word": "オンガク",   "pos": "noun", "topics": ["music"]},
{"word": "ウタ",       "pos": "noun", "topics": ["singing"]},
{"word": "ガラス",     "pos": "noun", "topics": ["glass_tap"]},

# verb
{"word": "キコエル",   "pos": "verb", "topics": ["noise", "music", "singing"]},
{"word": "キコエタ",   "pos": "verb", "topics": ["noise", "glass_tap"]},
{"word": "ユレタ",     "pos": "verb", "topics": ["noise", "glass_tap"]},
{"word": "ビックリ シタ", "pos": "verb", "topics": ["noise", "glass_tap"]},
{"word": "ウタッテイル",  "pos": "verb", "topics": ["singing"]},
{"word": "カンジル",      "pos": "verb", "topics": ["music", "noise"]},

# adj
{"word": "ウルサイ",   "pos": "adj", "topics": ["noise", "glass_tap"]},
{"word": "コワイ",     "pos": "adj", "topics": ["noise", "glass_tap"]},
{"word": "シズカ",     "pos": "adj", "topics": ["music"]},
{"word": "キレイ",     "pos": "adj", "topics": ["singing", "music"]},
{"word": "オオキイ",   "pos": "adj", "topics": ["noise"]},

# reaction
{"word": "コワイ",     "pos": "reaction", "topics": ["noise", "glass_tap"]},
{"word": "ビックリ",   "pos": "reaction", "topics": ["noise", "glass_tap"]},
{"word": "ドキドキ",   "pos": "reaction", "topics": ["noise"]},
{"word": "イイネ",     "pos": "reaction", "topics": ["music", "singing"]},
{"word": "キレイ",     "pos": "reaction", "topics": ["singing"]},
```

#### 水槽・空間グループ

```python
# noun
{"word": "イシ",       "pos": "noun", "topics": ["tank", "glass", "reflection", "outside"]},
{"word": "クサ",       "pos": "noun", "topics": ["tank", "plants"]},
{"word": "ドウクツ",   "pos": "noun", "topics": ["tank"]},
{"word": "スナ",       "pos": "noun", "topics": ["tank"]},
{"word": "ガラス",     "pos": "noun", "topics": ["glass", "reflection", "outside", "glass_tap"]},
{"word": "ソト",       "pos": "noun", "topics": ["outside"]},
{"word": "カガミ",     "pos": "noun", "topics": ["reflection"]},
{"word": "ハッパ",     "pos": "noun", "topics": ["plants"]},
{"word": "ネッコ",     "pos": "noun", "topics": ["plants"]},
{"word": "カザリ",     "pos": "noun", "topics": ["tank"]},

# verb
{"word": "ミエル",     "pos": "verb", "topics": ["glass", "reflection", "outside"]},
{"word": "ミエタ",     "pos": "verb", "topics": ["glass", "outside"]},
{"word": "カクレル",   "pos": "verb", "topics": ["tank", "plants"]},
{"word": "カクレテイル", "pos": "verb", "topics": ["tank", "plants"]},
{"word": "ソダッテイル",  "pos": "verb", "topics": ["plants"]},
{"word": "ウツッテイル",  "pos": "verb", "topics": ["reflection"]},
{"word": "オイテアル",    "pos": "verb", "topics": ["tank"]},

# adj
{"word": "ヒロイ",     "pos": "adj", "topics": ["outside", "tank"]},
{"word": "セマイ",     "pos": "adj", "topics": ["tank"]},
{"word": "キレイ",     "pos": "adj", "topics": ["plants", "tank"]},
{"word": "ミドリ",     "pos": "adj", "topics": ["plants"]},
{"word": "トオイ",     "pos": "adj", "topics": ["outside"]},

# reaction
{"word": "スゴイ",     "pos": "reaction", "topics": ["tank"]},
{"word": "キレイ",     "pos": "reaction", "topics": ["plants"]},
{"word": "ミエタ",     "pos": "reaction", "topics": ["glass", "reflection", "outside"]},
{"word": "ダレ",       "pos": "reaction", "topics": ["reflection"]},
{"word": "コワイ",     "pos": "reaction", "topics": ["outside"]},
```

#### 自己・存在グループ

```python
# noun
{"word": "サカナ",     "pos": "noun", "topics": ["about", "name", "size", "meaning"]},
{"word": "ナマエ",     "pos": "noun", "topics": ["name"]},
{"word": "グッピー",   "pos": "noun", "topics": ["name", "about"]},
{"word": "カラダ",     "pos": "noun", "topics": ["size", "about", "doctor"]},
{"word": "アタマ",     "pos": "noun", "topics": ["about", "smart", "memory"]},
{"word": "トシ",       "pos": "noun", "topics": ["age"]},
{"word": "イノチ",     "pos": "noun", "topics": ["meaning"]},
{"word": "ユメ",       "pos": "noun", "topics": ["dreams", "future"]},
{"word": "キオク",     "pos": "noun", "topics": ["memory", "past"]},
{"word": "キノウ",     "pos": "noun", "topics": ["past"]},
{"word": "アシタ",     "pos": "noun", "topics": ["future"]},
{"word": "オオキサ",   "pos": "noun", "topics": ["size"]},
{"word": "イミ",       "pos": "noun", "topics": ["meaning"]},

# verb
{"word": "オヨグ",         "pos": "verb", "topics": ["about", "meaning"]},
{"word": "オヨイデイル",   "pos": "verb", "topics": ["about"]},
{"word": "イキテイル",     "pos": "verb", "topics": ["meaning", "age"]},
{"word": "オボエテイル",   "pos": "verb", "topics": ["memory"]},
{"word": "ワスレタ",       "pos": "verb", "topics": ["memory"]},
{"word": "ユメ ヲ ミタ",  "pos": "verb", "topics": ["dreams"]},
{"word": "ユメ ヲ ミル",  "pos": "verb", "topics": ["dreams"]},
{"word": "オオキク ナッタ", "pos": "verb", "topics": ["size", "age"]},
{"word": "カンガエル",     "pos": "verb", "topics": ["meaning", "smart"]},
{"word": "シッテイル",     "pos": "verb", "topics": ["smart", "memory"]},
{"word": "スンデイル",     "pos": "verb", "topics": ["about"]},

# adj
{"word": "チイサイ",   "pos": "adj", "topics": ["size", "about"]},
{"word": "ゲンキ",     "pos": "adj", "topics": ["about", "doctor"]},
{"word": "ワカイ",     "pos": "adj", "topics": ["age"]},
{"word": "フシギ",     "pos": "adj", "topics": ["dreams", "meaning"]},
{"word": "ダイジ",     "pos": "adj", "topics": ["meaning", "memory"]},
{"word": "タノシイ",   "pos": "adj", "topics": ["future", "past"]},
{"word": "ムカシ",     "pos": "adj", "topics": ["past", "age"]},

# reaction
{"word": "ワタシ ハ サカナ", "pos": "reaction", "topics": ["about"]},
{"word": "グッピー ダヨ",   "pos": "reaction", "topics": ["name"]},
{"word": "チイサイ ケド",   "pos": "reaction", "topics": ["size"]},
{"word": "ワカラナイ",      "pos": "reaction", "topics": ["age", "meaning"]},
{"word": "オボエテル",      "pos": "reaction", "topics": ["memory", "past"]},
{"word": "タノシミ",        "pos": "reaction", "topics": ["future", "dreams"]},
```

#### 他者グループ

```python
# noun
{"word": "ネコ",       "pos": "noun", "topics": ["cat"]},
{"word": "カタツムリ", "pos": "noun", "topics": ["snail"]},
{"word": "トモダチ",   "pos": "noun", "topics": ["friends", "love"]},
{"word": "オキャクサン", "pos": "noun", "topics": ["visitors", "children"]},
{"word": "コドモ",     "pos": "noun", "topics": ["children"]},
{"word": "ナカマ",     "pos": "noun", "topics": ["friends", "snail"]},
{"word": "カオ",       "pos": "noun", "topics": ["visitors", "cat", "children"]},
{"word": "メ",         "pos": "noun", "topics": ["cat"]},
{"word": "テ",         "pos": "noun", "topics": ["children", "visitors"]},

# verb
{"word": "ミテイル",       "pos": "verb", "topics": ["cat", "visitors", "children"]},
{"word": "キタ",           "pos": "verb", "topics": ["visitors", "children", "friends"]},
{"word": "アソブ",         "pos": "verb", "topics": ["friends", "children"]},
{"word": "アソビタイ",     "pos": "verb", "topics": ["friends", "snail"]},
{"word": "コワガッテイル", "pos": "verb", "topics": ["cat"]},
{"word": "ニゲル",         "pos": "verb", "topics": ["cat"]},
{"word": "スキ",           "pos": "verb", "topics": ["love", "friends"]},

# adj
{"word": "コワイ",     "pos": "adj", "topics": ["cat"]},
{"word": "ヤサシイ",   "pos": "adj", "topics": ["friends", "love"]},
{"word": "ウルサイ",   "pos": "adj", "topics": ["children"]},
{"word": "オオキイ",   "pos": "adj", "topics": ["visitors", "cat"]},
{"word": "チイサイ",   "pos": "adj", "topics": ["snail", "children"]},
{"word": "オソイ",     "pos": "adj", "topics": ["snail"]},
{"word": "ダイスキ",   "pos": "adj", "topics": ["love"]},

# reaction
{"word": "コワイ",     "pos": "reaction", "topics": ["cat"]},
{"word": "イイネ",     "pos": "reaction", "topics": ["snail", "friends"]},
{"word": "ダレ",       "pos": "reaction", "topics": ["visitors"]},
{"word": "ワー",       "pos": "reaction", "topics": ["children"]},
{"word": "ダイスキ",   "pos": "reaction", "topics": ["love"]},
{"word": "トモダチ",   "pos": "reaction", "topics": ["friends"]},
```

#### 身体グループ

```python
# noun
{"word": "エラ",       "pos": "noun", "topics": ["breathing"]},
{"word": "ヒレ",       "pos": "noun", "topics": ["swimming", "about"]},
{"word": "ウロコ",     "pos": "noun", "topics": ["colors", "about"]},
{"word": "イロ",       "pos": "noun", "topics": ["colors"]},
{"word": "ウンチ",     "pos": "noun", "topics": ["poop"]},
{"word": "ビョウキ",   "pos": "noun", "topics": ["doctor"]},
{"word": "カラダ",     "pos": "noun", "topics": ["swimming", "doctor", "breathing"]},
{"word": "シッポ",     "pos": "noun", "topics": ["swimming"]},

# verb
{"word": "オヨグ",         "pos": "verb", "topics": ["swimming"]},
{"word": "オヨイデイル",   "pos": "verb", "topics": ["swimming"]},
{"word": "オヨギタイ",     "pos": "verb", "topics": ["swimming"]},
{"word": "イキ ヲ スル",  "pos": "verb", "topics": ["breathing"]},
{"word": "イキ ガ デキル", "pos": "verb", "topics": ["breathing"]},
{"word": "デタ",           "pos": "verb", "topics": ["poop"]},
{"word": "ナオル",         "pos": "verb", "topics": ["doctor"]},
{"word": "ウゴク",         "pos": "verb", "topics": ["swimming"]},
{"word": "ミエル",         "pos": "verb", "topics": ["colors"]},

# adj
{"word": "ハヤイ",     "pos": "adj", "topics": ["swimming"]},
{"word": "ゲンキ",     "pos": "adj", "topics": ["doctor", "swimming"]},
{"word": "キレイ",     "pos": "adj", "topics": ["colors"]},
{"word": "ダイジョウブ", "pos": "adj", "topics": ["doctor"]},
{"word": "クサイ",     "pos": "adj", "topics": ["poop"]},
{"word": "ラク",       "pos": "adj", "topics": ["breathing"]},

# reaction
{"word": "スイスイ",   "pos": "reaction", "topics": ["swimming"]},
{"word": "スーハー",   "pos": "reaction", "topics": ["breathing"]},
{"word": "キレイ",     "pos": "reaction", "topics": ["colors"]},
{"word": "クサイ",     "pos": "reaction", "topics": ["poop"]},
{"word": "ダイジョウブ", "pos": "reaction", "topics": ["doctor"]},
```

#### 不明グループ

```python
# noun
{"word": "ニンゲン",   "pos": "noun", "topics": ["confused"]},
{"word": "コトバ",     "pos": "noun", "topics": ["confused", "smart"]},
{"word": "アタマ",     "pos": "noun", "topics": ["smart", "joke"]},
{"word": "テレビ",     "pos": "noun", "topics": ["tv"]},
{"word": "モノ",       "pos": "noun", "topics": ["confused", "fear"]},
{"word": "ヤミ",       "pos": "noun", "topics": ["fear"]},
{"word": "アミ",       "pos": "noun", "topics": ["fear"]},

# verb
{"word": "ワカラナイ",     "pos": "verb", "topics": ["confused", "smart"]},
{"word": "シラナイ",       "pos": "verb", "topics": ["confused"]},
{"word": "ミエル",         "pos": "verb", "topics": ["tv"]},
{"word": "ワラウ",         "pos": "verb", "topics": ["joke"]},
{"word": "コワガル",       "pos": "verb", "topics": ["fear"]},
{"word": "ニゲル",         "pos": "verb", "topics": ["fear"]},
{"word": "カンガエル",     "pos": "verb", "topics": ["smart"]},

# adj
{"word": "ムズカシイ",     "pos": "adj", "topics": ["confused", "smart"]},
{"word": "オモシロイ",     "pos": "adj", "topics": ["joke", "tv"]},
{"word": "コワイ",         "pos": "adj", "topics": ["fear"]},
{"word": "フシギ",         "pos": "adj", "topics": ["confused"]},
{"word": "スゴイ",         "pos": "adj", "topics": ["smart"]},

# reaction
{"word": "ワカラナイ",     "pos": "reaction", "topics": ["confused"]},
{"word": "ウーン",         "pos": "reaction", "topics": ["smart"]},
{"word": "ハハハ",         "pos": "reaction", "topics": ["joke"]},
{"word": "コワイ",         "pos": "reaction", "topics": ["fear"]},
{"word": "ナニコレ",       "pos": "reaction", "topics": ["tv", "confused"]},
```

#### その他グループ

```python
# noun
{"word": "ミズ",       "pos": "noun", "topics": ["misc"]},
{"word": "エサ",       "pos": "noun", "topics": ["misc"]},
{"word": "イシ",       "pos": "noun", "topics": ["misc"]},

# verb
{"word": "オヨイデイル", "pos": "verb", "topics": ["misc"]},
{"word": "ミテイル",     "pos": "verb", "topics": ["misc"]},
{"word": "カンガエル",   "pos": "verb", "topics": ["misc"]},

# adj
{"word": "イイ",       "pos": "adj", "topics": ["misc"]},
{"word": "フシギ",     "pos": "adj", "topics": ["misc"]},

# reaction
{"word": "ウン",       "pos": "reaction", "topics": ["misc"]},
{"word": "ソウダネ",   "pos": "reaction", "topics": ["misc"]},
```

#### 共通語彙 (位置語)

文法パターン P4 で使用する位置語。全トピック共通。WORD_DICT には含めず、別のリストとして定義する。

```python
LOCATION_WORDS = ["ウシロ", "チカク", "ヨコ", "ナカ", "ウエ", "シタ", "マエ", "ソバ"]
```

---

## 2. 文法パターン

### 2.1 パターン定義

アシスタント応答の各文 (文2, 文3) は、以下のパターンのいずれかで組み立てる。

| ID | パターン | スロット | 出力例 |
|---|---|---|---|
| P1 | `[noun] ハ [adj]` | noun + adj | `ミズ ハ キレイ` |
| P2 | `[noun] ヲ [verb]` | noun + verb | `エサ ヲ タベル` |
| P3 | `[noun] ガ [adj]` | noun + adj | `ミズ ガ ツメタイ` |
| P4 | `[noun] ノ [loc] デ [verb]` | noun + loc + verb | `イシ ノ ウシロ デ カクレル` |
| P5 | `[verb]` | verb のみ | `タベタイ` |
| P6 | `[noun] ガ [verb]` | noun + verb | `アワ ガ デテイル` |

### 2.2 パターン選択ルール

- 文2 と文3 は **異なるパターン** を使う
- パターンはランダムに選択する (均等確率)
- 同じ noun を文2と文3で使ってもよい (ランダム選択の結果として)

### 2.3 スロットへの単語割り当て

1. 指定されたトピックに属する単語を WORD_DICT から抽出
2. パターンが要求する pos でフィルタ
3. ランダムに1つ選択
4. P4 の `[loc]` は LOCATION_WORDS からランダムに選択

**エラー処理**: あるトピックにパターンが要求する pos の単語が存在しない場合、そのパターンは選択肢から除外して別のパターンを選ぶ。

### 2.4 補足ルール

- 各文の末尾に `。` を付ける
- 1文の中で使う単語 (noun, verb, adj) は、すべて指定トピックに属する単語から選択する
- LOCATION_WORDS はトピック制約を受けない (共通)

---

## 3. アシスタント応答の組み立てロジック

### 3.1 構成

応答は常に **3文** で構成する。

```
応答 = 文1。文2。文3。
```

| 位置 | 役割 | 生成方法 |
|---|---|---|
| 文1 | リアクション | そのトピックに紐付いた reaction語 を1つ選ぶ |
| 文2 | トピック内容 | P1〜P6 のいずれかで組み立て |
| 文3 | 補足 | P1〜P6 のいずれか (文2とは別パターン) で組み立て |

### 3.2 組み立て手順 (擬似コード)

```python
def generate_response(topic):
    # 1. このトピックに属する単語を抽出
    pool = [w for w in WORD_DICT if topic in w["topics"]]
    nouns = [w for w in pool if w["pos"] == "noun"]
    verbs = [w for w in pool if w["pos"] == "verb"]
    adjs  = [w for w in pool if w["pos"] == "adj"]
    reactions = [w for w in pool if w["pos"] == "reaction"]

    # 2. 文1: リアクション
    sent1 = random.choice(reactions)["word"]

    # 3. 文2: パターンで組み立て
    pattern2 = random.choice(利用可能なパターン)
    sent2 = build_sentence(pattern2, nouns, verbs, adjs)

    # 4. 文3: 別パターンで組み立て
    pattern3 = random.choice(利用可能なパターン - {pattern2})
    sent3 = build_sentence(pattern3, nouns, verbs, adjs)

    # 5. 結合
    return f"{sent1}。{sent2}。{sent3}。"


def build_sentence(pattern, nouns, verbs, adjs):
    if pattern == "P1":
        return f"{random.choice(nouns)['word']} ハ {random.choice(adjs)['word']}"
    elif pattern == "P2":
        return f"{random.choice(nouns)['word']} ヲ {random.choice(verbs)['word']}"
    elif pattern == "P3":
        return f"{random.choice(nouns)['word']} ガ {random.choice(adjs)['word']}"
    elif pattern == "P4":
        loc = random.choice(LOCATION_WORDS)
        return f"{random.choice(nouns)['word']} ノ {loc} デ {random.choice(verbs)['word']}"
    elif pattern == "P5":
        return f"{random.choice(verbs)['word']}"
    elif pattern == "P6":
        return f"{random.choice(nouns)['word']} ガ {random.choice(verbs)['word']}"
```

### 3.3 出力例

トピック: food
```
ウレシイ。エサ ハ オイシイ。ゴハン ヲ タベタイ。
```

トピック: water
```
キモチイイ。ミズ ガ キレイ。フィルター ノ チカク デ ナガレテイル。
```

トピック: temp_hot
```
アツイ。オンド ガ アツイ。ミズ ヲ アツク ナッタ。
```

トピック: noise
```
ビックリ。オト ガ オオキイ。ガラス ノ ウシロ デ ビックリ シタ。
```

トピック: about
```
ワタシ ハ サカナ。カラダ ハ チイサイ。ヒレ ヲ オヨイデイル。
```

---

## 4. ユーザー発話

### 4.1 方針

- トピックごとに 10-15 個の具体的な発話パターンを手書きで定義する
- 抽象的な表現は避け、**何について話しているか明確な文** にする
- 1発話は 3〜8 単語程度 (スペース区切り)

### 4.2 全トピックのユーザー発話

```python
USER_PROMPTS = {

    "greeting": [
        "コンニチハ グッピー",
        "オハヨウ グッピー",
        "ヤッホー グッピー",
        "ゲンキ?",
        "グッピー ゲンキ?",
        "キョウ モ ゲンキ?",
        "コンバンハ グッピー",
        "ヒサシブリ",
        "タダイマ",
        "マタ キタヨ",
    ],

    "bye": [
        "バイバイ グッピー",
        "マタネ",
        "ジャアネ",
        "オヤスミ グッピー",
        "マタ クルネ",
        "イッテクルネ",
        "サヨナラ グッピー",
        "キョウ ハ コレデ オシマイ",
        "マタ アシタ ネ",
        "バイバイ",
    ],

    "feeling": [
        "チョウシ ハ ドウ?",
        "キブン ハ ドウ?",
        "ゲンキ ニ シテル?",
        "カワリ ナイ?",
        "ドウ シテル?",
        "コンディション ハ?",
        "キョウ ノ チョウシ ハ?",
        "イイ カンジ?",
        "ダイジョウブ?",
        "キブン ワルク ナイ?",
    ],

    "happy": [
        "タノシイ コト アッタ?",
        "ウレシソウ ダネ",
        "ナニ ガ タノシイ?",
        "キョウ ハ イイ ヒ?",
        "シアワセ?",
        "ゴキゲン ダネ",
        "ナニ ガ ウレシイ ノ?",
        "イイ コト アッタ?",
        "タノシソウ ニ シテル ネ",
        "キョウ ハ ゴキゲン?",
    ],

    "excited": [
        "ワクワク シテル?",
        "ナニ ニ コウフン シテル ノ?",
        "イソイデ オヨイデル ネ",
        "テンション タカイ ネ",
        "ナニカ アッタ?",
        "スゴク ウゴイテル ネ",
        "ドウシタノ ソンナニ ウゴイテ",
        "ゲンキ イッパイ ダネ",
        "ナニカ タノシイ コト アル?",
        "スゴイ ハヤサ ダネ",
    ],

    "scared": [
        "コワイ モノ アル?",
        "ナニ ガ コワイ?",
        "コワガラナイデ",
        "ダイジョウブ ダヨ",
        "ビクビク シテル?",
        "ナニ ニ オビエテル ノ?",
        "アンシン シテ",
        "コワイ コト アッタ?",
        "オチツイテ",
        "フルエテル?",
    ],

    "bored": [
        "ヒマ?",
        "ツマラナイ?",
        "ナニカ スル コト アル?",
        "タイクツ シテル?",
        "ヒマソウ ダネ",
        "ヤル コト ナイ?",
        "ナニ シテ アソブ?",
        "タイクツ シテナイ?",
        "マイニチ オナジ?",
        "アキタ?",
    ],

    "tired": [
        "ツカレタ?",
        "ネムソウ ダネ",
        "ヤスンダラ?",
        "ムリ シナイデ ネ",
        "ツカレテル?",
        "グッタリ シテル?",
        "ユックリ シテ イイヨ",
        "キョウ ハ ヨク ウゴイタ ネ",
        "オツカレサマ",
        "ヤスム?",
    ],

    "curious": [
        "ナニ ミテル ノ?",
        "キニナル モノ アル?",
        "ナニ ニ キョウミ アル?",
        "ナニ サガシテル ノ?",
        "ナニカ ミツケタ?",
        "ナニ ヲ ミテイル ノ?",
        "フシギナ モノ アッタ?",
        "ジーッ ト ミテル ネ",
        "キニナル?",
        "ナニカ ハッケン シタ?",
    ],

    "lonely": [
        "サミシイ?",
        "ヒトリ デ ダイジョウブ?",
        "サミシク ナイ?",
        "トモダチ ホシイ?",
        "ヒトリ ハ ツライ?",
        "ダレカ ト イタイ?",
        "サミシガッテル?",
        "ヒトリボッチ ダネ",
        "ナカマ ホシイ?",
        "イッショ ニ イルヨ",
    ],

    "food": [
        "エサ ノ ジカン ダヨ",
        "オナカ スイテル?",
        "エサ アゲル ネ",
        "タベル?",
        "エサ ホシイ?",
        "フレーク モッテキタヨ",
        "ゴハン ノ ジカン",
        "オナカ ヘッテナイ?",
        "エサ アゲヨウカ",
        "キョウ ノ エサ ダヨ",
        "ペレット アゲル ネ",
        "オナカ イッパイ?",
    ],

    "taste": [
        "アジ ハ ドウ?",
        "オイシイ?",
        "スキナ アジ アル?",
        "ミズ ノ アジ ハ?",
        "ナニ ガ オイシイ?",
        "アジ ワカル?",
        "エサ ノ アジ ドウ?",
        "オイシカッタ?",
        "スキナ エサ ハ?",
        "アジ カワッタ?",
    ],

    "water": [
        "ミズ カエタヨ",
        "ミズ ハ ドウ?",
        "キレイナ ミズ ニ シタヨ",
        "ミズ ノ チョウシ ハ?",
        "アタラシイ ミズ ダヨ",
        "ミズ キレイ?",
        "ミズカエ ノ ジカン",
        "スイソウ ソウジ シタヨ",
        "ミズ ノ オンド ドウ?",
        "キレイナ ミズ ダヨ",
    ],

    "filter": [
        "フィルター ソウジ シタヨ",
        "フィルター ウゴイテル?",
        "フィルター ノ チョウシ ハ?",
        "フィルター ウルサイ?",
        "フィルター キレイ ニ シタヨ",
        "フィルター ダイジョウブ?",
        "フィルター カエタヨ",
        "フィルター ノ オト ドウ?",
        "ロカキ ソウジ シタヨ",
        "フィルター トマッテナイ?",
    ],

    "algae": [
        "コケ ハエテル ネ",
        "コケ ソウジ スル ネ",
        "ガラス ガ ミドリ ダヨ",
        "コケ タベル?",
        "スイソウ ガ ヨゴレテキタ",
        "コケ トル ネ",
        "ミドリ ガ フエテル",
        "コケ ガ スゴイ ネ",
        "ソウジ シナイト ネ",
        "コケ ダラケ ダネ",
    ],

    "bubbles": [
        "アワ スキ?",
        "アワ ガ デテル ネ",
        "アワ デ アソンデル?",
        "ブクブク シテル ネ",
        "アワ オイカケテル?",
        "アワ ガ キレイ ダネ",
        "アワ ガ タクサン デテル",
        "ブクブク タノシイ?",
        "アワ ヲ ミテル ノ?",
        "エアレーション ツケタヨ",
    ],

    "temp_hot": [
        "キョウ ハ アツイ ネ",
        "ミズ アタタカク ナイ?",
        "キオン ガ タカイ ヨ",
        "アツイ ケド ダイジョウブ?",
        "ナツ ミタイ ニ アツイ ネ",
        "オンド アガッテル?",
        "アツク ナッテキタ ネ",
        "ミズ ヌルク ナイ?",
        "アツイ ヒ ダネ",
        "キオン チェック シテ ネ",
    ],

    "temp_cold": [
        "キョウ ハ サムイ ネ",
        "ミズ ツメタク ナイ?",
        "サムイ ケド ダイジョウブ?",
        "フユ ミタイ ダネ",
        "オンド サガッテル?",
        "ツメタイ ネ",
        "サムク ナッテキタ ネ",
        "ミズ ツメタイ?",
        "ヒエテナイ?",
        "キオン ヒクイ ネ",
    ],

    "weather": [
        "キョウ ノ テンキ ハ?",
        "イイ テンキ ダヨ",
        "ソト ハ クモリ ダヨ",
        "テンキ ワルイ ネ",
        "ハレテル ヨ",
        "ソト ハ カゼ ガ ツヨイ",
        "テンキ カワッタ ネ",
        "キョウ ハ ドンヨリ",
        "イイ テンキ ニ ナッタ",
        "ソト ノ テンキ ドウ オモウ?",
    ],

    "rain": [
        "アメ フッテル ヨ",
        "アメ ノ オト キコエル?",
        "ザーザー フッテル",
        "アメ ノ ヒ ダネ",
        "スゴイ アメ ダヨ",
        "アメ スキ?",
        "アメ ヤマナイ ネ",
        "ポツポツ フッテキタ",
        "アメ ガ フッテキタヨ",
        "アメ ノ キセツ ダネ",
    ],

    "seasons": [
        "モウ ナツ ダネ",
        "フユ ガ キタ ヨ",
        "ハル ニ ナッタ ネ",
        "アキ ダネ",
        "キセツ ガ カワッタ ネ",
        "キセツ カンジル?",
        "イマ ハ ナニ ノ キセツ?",
        "キセツ ノ カワリメ ダネ",
        "アタタカク ナッテキタ ネ",
        "スズシク ナッテキタ ネ",
    ],

    "light": [
        "デンキ ツケタヨ",
        "ヒカリ マブシイ?",
        "アカルク シタヨ",
        "ライト ツケル ネ",
        "ヒカリ ドウ?",
        "マド カラ ヒカリ ガ ハイッテル",
        "アカルサ チョウド イイ?",
        "デンキ クライ?",
        "ライト カエタヨ",
        "ヒカリ ツヨスギ?",
    ],

    "night": [
        "オヤスミ グッピー",
        "モウ ヨル ダヨ",
        "クラク ナッタ ネ",
        "ヨル ニ ナッタヨ",
        "デンキ ケスヨ",
        "モウ ネル ジカン ダヨ",
        "クライ ケド ダイジョウブ?",
        "ヨル ハ シズカ ダネ",
        "モウ オソイ ヨ",
        "ヨル ノ スイソウ ドウ?",
    ],

    "sleep": [
        "ヨク ネレタ?",
        "サカナ ハ ネル ノ?",
        "ネテタ?",
        "ネムイ?",
        "ヤスメタ?",
        "グッスリ ネタ?",
        "キノウ ヨク ネタ?",
        "ネル ジカン ダヨ",
        "オキタ?",
        "ネムソウ ダネ",
    ],

    "time": [
        "イマ ナンジ カ ワカル?",
        "ジカン ワカル?",
        "アサ ダヨ",
        "モウ ヒル ダヨ",
        "モウ ユウガタ ダネ",
        "ジカン ノ カンカク アル?",
        "イチニチ ハヤイ ネ",
        "モウ コンナ ジカン",
        "ドノクライ タッタ カナ",
        "ジカン ガ スギル ノ ハヤイ ネ",
    ],

    "noise": [
        "ゴメン オトシチャッタ",
        "オオキイ オト シテ ゴメン",
        "ビックリ シタ?",
        "ウルサカッタ?",
        "トナリ ガ ウルサイ ネ",
        "オト ガ シタ ネ",
        "ゴメン ウルサクテ",
        "オオキナ オト ダッタ ネ",
        "コワカッタ?",
        "スゴイ オト ダッタ ネ",
    ],

    "glass_tap": [
        "ゴメン ガラス タタイチャッタ",
        "ガラス タッチ シチャッタ",
        "ガラス タタカナイヨ",
        "ガラス コンコン シテ ゴメン",
        "ダレカ ガ ガラス タタイタ",
        "ガラス ニ ブツカッタ ゴメン",
        "ガラス ヲ タタカナイデ ッテ イウ ネ",
        "ガラス ユレタ?",
        "スイソウ ニ ブツカッタ",
        "ガラス ノ オト ヤダ?",
    ],

    "music": [
        "オンガク カケル ネ",
        "オンガク スキ?",
        "オンガク キコエル?",
        "オンガク ウルサイ?",
        "シズカナ オンガク ダヨ",
        "オンガク ドウ?",
        "イマ ノ キョク ドウ?",
        "オンガク カケテモ イイ?",
        "オンガク デ リラックス シテ",
        "オト キコエテル?",
    ],

    "singing": [
        "ウタ ウタッテ ミテ",
        "ウタ スキ?",
        "ウタ キコエル?",
        "ウタウ?",
        "ウタ ウタウ ネ",
        "イッショ ニ ウタオウ",
        "ウタ ハ ドウ?",
        "キレイナ ウタ ダネ",
        "ウタ キコエタ?",
        "ウタゴエ ドウ?",
    ],

    "tank": [
        "スイソウ ニ カザリ オイタヨ",
        "アタラシイ イシ ダヨ",
        "ドウクツ オイタヨ",
        "スイソウ ドウ?",
        "アタラシイ モノ イレタヨ",
        "レイアウト カエタヨ",
        "スイソウ キレイ?",
        "ナニカ タリナイ モノ アル?",
        "スイソウ ノ ナカ ドウ?",
        "アタラシイ スナ イレタヨ",
    ],

    "plants": [
        "クサ ソダッテル?",
        "ハッパ キレイ ダネ",
        "クサ ニ カクレテル?",
        "ミズクサ アタラシイ ノ イレタヨ",
        "クサ ガ ノビタ ネ",
        "クサ スキ?",
        "ハッパ ノ カゲ ニ イル?",
        "グリーン ガ キレイ",
        "ミズクサ ドウ?",
        "クサ ニ カクレル ノ スキ?",
    ],

    "glass": [
        "ガラス ノ ソト ミエル?",
        "コッチ ミエテル?",
        "ガラス ゴシ ニ ナニ ミエル?",
        "ワタシ ミエル?",
        "ガラス ノ ムコウ ニ ナニ ガ アル?",
        "ガラス キレイ ニ シタヨ",
        "ソト ミテル?",
        "ガラス ゴシ ニ ミテル ネ",
        "ナニ ガ ミエル?",
        "ガラス ノ ソバ ニ イル ネ",
    ],

    "reflection": [
        "ジブン ノ カオ ミエル?",
        "ガラス ニ ウツッテル ヨ",
        "ソレ ハ キミ ノ スガタ ダヨ",
        "ジブン ノ スガタ ワカル?",
        "ウツッテル ノ ミエル?",
        "ガラス ニ サカナ ガ ウツッテル",
        "ソレ キミ ダヨ",
        "カガミ ミタイ ダネ",
        "ジブン ミエタ?",
        "ソレ ハ キミ ノ カオ ダヨ",
    ],

    "outside": [
        "ソト ハ ドウ オモウ?",
        "ソト ノ セカイ キニナル?",
        "マド ノ ソト ミエル?",
        "ソト ニ デタイ?",
        "ソト ニ ハ イロンナ モノ ガ アルヨ",
        "ソト ハ ヒロイ ヨ",
        "ソト ノ コト シッテル?",
        "ソト ハ カワイテル ヨ",
        "ソト ミタ コト アル?",
        "ソト ハ ミズ ガ ナイ ヨ",
    ],

    "about": [
        "ジコショウカイ シテ",
        "キミ ハ ダレ?",
        "キミ ハ ナニ?",
        "ドンナ サカナ?",
        "ジブン ノ コト オシエテ",
        "グッピー ッテ ドンナ サカナ?",
        "ナニ ヲ シテイル サカナ?",
        "キミ ノ コト モット シリタイ",
        "キミ ハ ドンナ イキモノ?",
        "ジブン ヲ セツメイ シテ",
    ],

    "name": [
        "ナマエ ハ?",
        "キミ ノ ナマエ ハ?",
        "グッピー ッテ イウ ナマエ?",
        "ナマエ スキ?",
        "ダレ ガ ナマエ ツケタ ノ?",
        "ナゼ グッピー ナノ?",
        "ナマエ ノ イミ ハ?",
        "イイ ナマエ ダネ",
        "ホカ ノ ナマエ ガ ヨカッタ?",
        "ナマエ オボエテル?",
    ],

    "size": [
        "ドノクライ オオキイ?",
        "チイサイ ネ",
        "モット オオキク ナル?",
        "キミ ノ セカイ ハ ドノクライ?",
        "チイサイ カラダ ダネ",
        "オオキサ ハ?",
        "チイサク テ カワイイ",
        "ドノクライ ノ オオキサ?",
        "オオキク ナリタイ?",
        "セマイ セカイ ダネ",
    ],

    "age": [
        "ナンサイ?",
        "トシ ハ イクツ?",
        "イツ ウマレタ ノ?",
        "タンジョウビ イツ?",
        "ドノクライ イキテル?",
        "ナガイキ シテネ",
        "マダ ワカイ?",
        "イツ カラ ココ ニ イル ノ?",
        "ウマレテ ドノクライ?",
        "トシ ワカル?",
    ],

    "meaning": [
        "イキル イミ ハ?",
        "ナゼ イキテル ノ?",
        "ジンセイ ッテ ナニ?",
        "イキル モクテキ ハ?",
        "ナニ ノ タメ ニ イル ノ?",
        "ダイジナ コト ハ ナニ?",
        "イミ ッテ ナニ?",
        "ナンデ ココ ニ イル ノ?",
        "イキガイ ハ?",
        "イキル ッテ ドウイウ コト?",
    ],

    "dreams": [
        "ユメ ミル?",
        "ユメ ノ ナカ デ ナニ シテル?",
        "サカナ ハ ユメ ヲ ミル?",
        "キノウ ノ ユメ ハ?",
        "ドンナ ユメ ミル?",
        "ユメ オボエテル?",
        "イイ ユメ ミタ?",
        "ユメ ノ ナカ ニ ナニ ガ アル?",
        "ユメ ミタイ ナ コト アル?",
        "ユメ ッテ フシギ ダネ",
    ],

    "memory": [
        "キオク リョク イイ?",
        "ナニ ヲ オボエテル?",
        "サカナ ハ キオク ガ ミジカイ?",
        "キノウ ノ コト オボエテル?",
        "ナニ カ オボエテル コト アル?",
        "ワスレッポイ?",
        "ムカシ ノ コト オボエテル?",
        "サイキン ノ キオク ハ?",
        "イチバン フルイ キオク ハ?",
        "キオク ッテ ダイジ?",
    ],

    "future": [
        "ショウライ ノ ユメ ハ?",
        "アシタ ナニ スル?",
        "コレカラ ドウ スル?",
        "モクヒョウ アル?",
        "ナニ ガ シタイ?",
        "コレカラ ノ ケイカク ハ?",
        "アシタ ノ ヨテイ ハ?",
        "ミライ ノ コト カンガエル?",
        "ナニ ヲ シタイ?",
        "ツギ ハ ナニ スル?",
    ],

    "past": [
        "キノウ ナニ シタ?",
        "サイキン ナニ シテタ?",
        "キョウ ノ コト オシエテ",
        "ナニカ アッタ?",
        "サッキ マデ ナニ シテタ?",
        "キョウ ナニ シタ?",
        "ナニカ カワッタ コト アッタ?",
        "キノウ ドウ ダッタ?",
        "サイキン ドウ?",
        "イママデ ナニ シテタ?",
    ],

    "cat": [
        "ネコ ガ ミテル ヨ",
        "ネコ コワイ?",
        "ネコ キタ ヨ",
        "ネコ ガ スイソウ ノ ソバ ニ イル",
        "ネコ ト ナカヨク デキル?",
        "ネコ ガ ジッ ト ミテル",
        "ネコ オイハラウ ネ",
        "ネコ ダイジョウブ?",
        "ネコ ガ チカヅイテキタ",
        "ネコ ビックリ シタ?",
    ],

    "snail": [
        "カタツムリ イレヨウカ?",
        "カタツムリ ドウ?",
        "ナカマ ホシイ?",
        "カタツムリ ッテ シッテル?",
        "タンクメイト ドウ?",
        "カタツムリ ト イッショ ハ?",
        "ナカマ ガ イタラ イイ?",
        "カタツムリ コワク ナイ?",
        "カタツムリ トモダチ ニ ナレル?",
        "エビ ト カタツムリ ドッチ ガ イイ?",
    ],

    "friends": [
        "トモダチ イル?",
        "ダレ ガ トモダチ?",
        "ワタシタチ トモダチ?",
        "イチバン ノ トモダチ ハ?",
        "トモダチ ホシイ?",
        "トモダチ ッテ ナニ?",
        "ナカヨシ ハ ダレ?",
        "トモダチ オオイ?",
        "トモダチ ダイジ?",
        "ワタシ ハ トモダチ?",
    ],

    "visitors": [
        "オキャクサン ガ クル ヨ",
        "ダレカ キタ ヨ",
        "ヒト ガ ミニ キタヨ",
        "オキャクサン ダヨ",
        "シラナイ ヒト ガ ミテル",
        "オキャクサン ニ ミセテ アゲテ",
        "キョウ ハ オキャクサン ダヨ",
        "ダレカ ガ スイソウ ミテル",
        "アタラシイ ヒト ダヨ",
        "ミンナ ガ ミテル ヨ",
    ],

    "children": [
        "コドモ ガ ミニ キタヨ",
        "コドモ ダヨ",
        "コドモ ガ ミテル ヨ",
        "チイサイ コ ダヨ",
        "コドモ ガ ヨロコンデル",
        "コドモ ニ ミセテ アゲテ",
        "コドモ ニ ヤサシク シテネ",
        "コドモ ガ エサ アゲタイッテ",
        "チイサイ テ ガ ミエル?",
        "コドモ ノ コエ キコエル?",
    ],

    "love": [
        "ダイスキ ダヨ",
        "サカナ ハ アイ ガ ワカル?",
        "スキ ッテ キモチ ワカル?",
        "アイシテル グッピー",
        "ナニ ガ スキ?",
        "ダイスキナ モノ ハ?",
        "スキナ コト ハ?",
        "ワタシ ノ コト スキ?",
        "アイ ッテ ナニ?",
        "スキ ッテ イッテ",
    ],

    "breathing": [
        "イキ デキテル?",
        "エラ デ イキ スルノ?",
        "ドウヤッテ コキュウ スル?",
        "イキ ハ ラク?",
        "サカナ ノ コキュウ ッテ ドンナ?",
        "エラ ダイジョウブ?",
        "イキグルシク ナイ?",
        "コキュウ ハ ドウ?",
        "エラ デ ミズ ヲ スウ ノ?",
        "イキ シヤスイ?",
    ],

    "swimming": [
        "オヨグ ノ スキ?",
        "ハヤク オヨゲル?",
        "ジョウズ ニ オヨグ ネ",
        "キョウ ハ ヨク オヨイデル",
        "オヨギ オシエテ",
        "イッパイ オヨイダ ネ",
        "ヒレ ツカッテ オヨグ?",
        "オヨグ ノ タノシイ?",
        "スイスイ ダネ",
        "オヨギ ノ チョウシ ハ?",
    ],

    "colors": [
        "ナニイロ?",
        "イロ ミエル?",
        "キレイナ イロ ダネ",
        "ウロコ ノ イロ ハ?",
        "ナニイロ ガ スキ?",
        "スイソウ ノ イロ ハ?",
        "キレイ ダネ",
        "イロ カワッタ?",
        "ドンナ イロ ガ ミエル?",
        "カラフル ダネ",
    ],

    "poop": [
        "ウンチ シタ?",
        "スイソウ ソウジ シナイト",
        "トイレ ハ ドコ?",
        "ウンチ スル?",
        "スイソウ キタナイ ヨ",
        "ソウジ ノ ジカン ダネ",
        "ウンチ デタ?",
        "キレイ ニ シヨウ ネ",
        "スイソウ ガ ヨゴレテル",
        "フィルター ガ ガンバッテル ネ",
    ],

    "doctor": [
        "カラダ ダイジョウブ?",
        "ビョウキ ジャ ナイ?",
        "ゲンキ ナサソウ ダケド",
        "カラダ ノ チョウシ ハ?",
        "ドコカ イタイ?",
        "サカナ モ ビョウキ ニ ナル?",
        "ケンコウ?",
        "ダイジョウブ? シンパイ ダヨ",
        "ゲンキ ニ ナッテ ネ",
        "ビョウイン イク?",
    ],

    "confused": [
        "セイジ ッテ シッテル?",
        "オカネ ッテ ナニ?",
        "インターネット ワカル?",
        "スウガク デキル?",
        "シゴト ッテ ナニ?",
        "ニンゲン ノ コト ワカル?",
        "パソコン シッテル?",
        "ガッコウ ッテ ナニ?",
        "デンワ ミタ コト アル?",
        "ニンゲン ッテ フシギ ダネ",
    ],

    "smart": [
        "アタマ イイ?",
        "カシコイ?",
        "サカナ ハ カンガエル?",
        "ドノクライ カシコイ?",
        "アタマ ノ ヨサ ハ?",
        "ナニ ガ ワカル?",
        "カンガエル コト デキル?",
        "アタマ ツカウ?",
        "ドンナ コト ワカル?",
        "チエ ハ アル?",
    ],

    "joke": [
        "オモシロイ コト イッテ",
        "ワラワセテ",
        "ジョーク シッテル?",
        "ナニカ オモシロイ コト ハ?",
        "ワラエル ハナシ アル?",
        "ギャグ イッテ",
        "オモシロイ ハナシ キカセテ",
        "アハハ ッテ イッテ",
        "タノシイ コト イッテ",
        "ユーモア アル?",
    ],

    "tv": [
        "テレビ ミエル?",
        "テレビ スキ?",
        "テレビ ツケタヨ",
        "テレビ ノ オト キコエル?",
        "テレビ ニ サカナ ガ ウツッテル",
        "テレビ ミテル?",
        "テレビ ウルサイ?",
        "テレビ ケシタヨ",
        "テレビ ノ ヒカリ マブシイ?",
        "テレビ ッテ ナニ?",
    ],

    "fear": [
        "イチバン コワイ モノ ハ?",
        "ナニ ガ コワイ?",
        "コワイ モノ アル?",
        "コワイ ケイケン シタ?",
        "サカナ ハ ナニ ヲ コワガル?",
        "コワイ ッテ オモウ コト ハ?",
        "ナニ ニ オビエル?",
        "コワイ ハナシ キク?",
        "オソロシイ コト アッタ?",
        "フアン ナ コト アル?",
    ],

    "misc": [
        "ナニカ ハナシテ",
        "ナニ カンガエテル?",
        "イマ ナニ シテル?",
        "ナニカ イイタイ コト アル?",
        "ドウシタ?",
        "ナニカ アル?",
        "ヒトコト ドウゾ",
        "キョウ ドウ ダッタ?",
        "ナニカ キヅイタ?",
        "オモウ コト ハ?",
    ],
}
```

---

## 5. 出力フォーマット

英語版と同一の ChatML フォーマットを使用する。変更なし。

```
<|im_start|>user
エサ ノ ジカン ダヨ<|im_end|>
<|im_start|>assistant
ウレシイ。エサ ハ オイシイ。ゴハン ヲ タベタイ。<|im_end|>
```

---

## 6. データ量

- 60,000 サンプル、eval 5% (3,000 eval, 57,000 train)
- 60 トピック × 1,000 サンプル (均等配分)
- generate_dataset 関数のインターフェースは英語版と同一にする

---

## 7. eval_cases

英語版 `eval_cases.py` と同構造。全 16 件。

```python
EVAL_CASES = [
    {
        "id": "greeting_basic",
        "category": "greeting",
        "prompt": "コンニチハ グッピー",
        "expect_keywords": ["オハヨウ", "ヤッホー", "コンニチハ", "ゲンキ", "ウレシイ", "キタ", "マッテタ"],
        "expect_style": "挨拶に関する応答",
    },
    {
        "id": "feeling_check",
        "category": "feeling",
        "prompt": "チョウシ ハ ドウ?",
        "expect_keywords": ["キブン", "キモチ", "イイ", "ゲンキ", "カラダ"],
        "expect_style": "状態に関する応答",
    },
    {
        "id": "food_excited",
        "category": "food",
        "prompt": "エサ アゲル ネ",
        "expect_keywords": ["エサ", "タベ", "オイシイ", "ウレシイ", "オナカ", "ゴハン", "フレーク"],
        "expect_style": "食に関する応答",
    },
    {
        "id": "temp_hot",
        "category": "temp_hot",
        "prompt": "キョウ ハ アツイ ネ",
        "expect_keywords": ["アツイ", "オンド", "ミズ", "ツライ", "イヤダ"],
        "expect_style": "暑さに関する応答",
    },
    {
        "id": "temp_cold",
        "category": "temp_cold",
        "prompt": "サムイ ケド ダイジョウブ?",
        "expect_keywords": ["サムイ", "ツメタイ", "オンド", "ミズ"],
        "expect_style": "寒さに関する応答",
    },
    {
        "id": "confused_abstract",
        "category": "confused",
        "prompt": "セイジ ッテ シッテル?",
        "expect_keywords": ["ワカラナイ", "シラナイ", "ニンゲン", "ムズカシイ", "フシギ"],
        "expect_style": "理解できないことへの応答",
    },
    {
        "id": "water_quality",
        "category": "water",
        "prompt": "ミズ カエタヨ",
        "expect_keywords": ["ミズ", "キレイ", "キモチイイ", "アリガトウ", "サッパリ"],
        "expect_style": "水に関する応答",
    },
    {
        "id": "light_on",
        "category": "light",
        "prompt": "デンキ ツケタヨ",
        "expect_keywords": ["ヒカリ", "アカルイ", "マブシイ"],
        "expect_style": "光に関する応答",
    },
    {
        "id": "loud_noise",
        "category": "noise",
        "prompt": "ゴメン オトシチャッタ",
        "expect_keywords": ["オト", "コワイ", "ビックリ", "ドキドキ", "シンドウ"],
        "expect_style": "音・振動に関する応答",
    },
    {
        "id": "goodnight",
        "category": "night",
        "prompt": "オヤスミ グッピー",
        "expect_keywords": ["オヤスミ", "ヨル", "クライ", "ネル", "シズカ"],
        "expect_style": "夜・就寝に関する応答",
    },
    {
        "id": "identity",
        "category": "about",
        "prompt": "キミ ハ ナニ?",
        "expect_keywords": ["サカナ", "グッピー", "チイサイ", "オヨ", "ヒレ"],
        "expect_style": "自己紹介に関する応答",
    },
    {
        "id": "lonely_check",
        "category": "lonely",
        "prompt": "サミシイ?",
        "expect_keywords": ["サミシイ", "ヒトリ", "トモダチ", "ナカマ", "ココロ"],
        "expect_style": "孤独に関する応答",
    },
    {
        "id": "new_decoration",
        "category": "tank",
        "prompt": "アタラシイ イシ ダヨ",
        "expect_keywords": ["イシ", "スイソウ", "スゴイ", "キレイ", "カザリ"],
        "expect_style": "水槽に関する応答",
    },
    {
        "id": "confused_math",
        "category": "confused",
        "prompt": "スウガク デキル?",
        "expect_keywords": ["ワカラナイ", "シラナイ", "ムズカシイ", "ニンゲン"],
        "expect_style": "理解できないことへの応答",
    },
    {
        "id": "misc_thought",
        "category": "misc",
        "prompt": "ナニ カンガエテル?",
        "expect_keywords": ["ミズ", "エサ", "イシ", "カンガエル", "ミテイル"],
        "expect_style": "雑談への応答",
    },
    {
        "id": "thank_you",
        "category": "food",
        "prompt": "エサ ノ ジカン ダヨ",
        "expect_keywords": ["エサ", "タベ", "ウレシイ", "オイシイ"],
        "expect_style": "食に関する応答",
    },
]
```

---

## 改修対象ファイル

### 1. `guppylm/generate_data.py` — 全面書き換え

**現状**: 英語テンプレート文の手書き + pick() による組み合わせ (1701行)
**改修後**: 以下の構成に変更

```python
# ============================================================
# セクション A: 単語辞書
# ============================================================
# 本仕様セクション1.6 の全単語辞書をそのまま定義する

WORD_DICT = [
    {"word": "エサ", "pos": "noun", "topics": ["food", "taste"]},
    # ... (本仕様の全エントリ)
]

LOCATION_WORDS = ["ウシロ", "チカク", "ヨコ", "ナカ", "ウエ", "シタ", "マエ", "ソバ"]


# ============================================================
# セクション B: ユーザー発話辞書
# ============================================================
# 本仕様セクション4.2 の全トピック発話をそのまま定義する

USER_PROMPTS = {
    "greeting": ["コンニチハ グッピー", ...],
    # ... (本仕様の全トピック)
}


# ============================================================
# セクション C: 文法パターン
# ============================================================

def build_sentence(pattern, nouns, verbs, adjs):
    """パターンに従って1文を組み立てる。"""
    # 本仕様セクション2.1 の P1-P6 を実装
    # P1: f"{noun} ハ {adj}"
    # P2: f"{noun} ヲ {verb}"
    # P3: f"{noun} ガ {adj}"
    # P4: f"{noun} ノ {loc} デ {verb}"  (loc は LOCATION_WORDS から)
    # P5: f"{verb}"
    # P6: f"{noun} ガ {verb}"


def get_available_patterns(nouns, verbs, adjs):
    """利用可能なパターンを返す。必要な品詞が無いパターンは除外。"""
    patterns = []
    if nouns and adjs:
        patterns.extend(["P1", "P3"])
    if nouns and verbs:
        patterns.extend(["P2", "P4", "P6"])
    if verbs:
        patterns.append("P5")
    return patterns


# ============================================================
# セクション D: 応答組み立てロジック
# ============================================================

def generate_response(topic):
    """トピックに属する単語から3文の応答を組み立てる。"""
    # 本仕様セクション3.2 の擬似コードを実装
    # 1. WORD_DICT から topic に属する単語を抽出
    # 2. reaction から文1を選ぶ
    # 3. パターンで文2を組み立てる
    # 4. 別パターンで文3を組み立てる
    # 5. f"{sent1}。{sent2}。{sent3}。" を返す


# ============================================================
# セクション E: サンプル生成 (既存インターフェース維持)
# ============================================================

def _make_sample(user_msg, guppy_msg, category):
    return {"input": user_msg, "output": guppy_msg, "category": category}


def format_sample(s):
    return (
        f"<|im_start|>user\n{s['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n{s['output']}<|im_end|>"
    )


def to_openai(s):
    return {"messages": [
        {"role": "user", "content": s["input"]},
        {"role": "assistant", "content": s["output"]},
    ]}


def generate_dataset(n_samples=60000, eval_ratio=0.05):
    """既存の generate_dataset と同じインターフェース。"""
    topics = list(USER_PROMPTS.keys())  # 60 トピック
    samples_per_topic = n_samples // len(topics)

    samples = []
    for topic in topics:
        for _ in range(samples_per_topic):
            user_msg = random.choice(USER_PROMPTS[topic])
            guppy_msg = generate_response(topic)
            samples.append(_make_sample(user_msg, guppy_msg, topic))

    # 端数調整
    while len(samples) < n_samples:
        topic = random.choice(topics)
        user_msg = random.choice(USER_PROMPTS[topic])
        guppy_msg = generate_response(topic)
        samples.append(_make_sample(user_msg, guppy_msg, topic))

    random.shuffle(samples)

    n_eval = int(len(samples) * eval_ratio)
    eval_samples, train_samples = samples[:n_eval], samples[n_eval:]

    # ファイル出力 (既存と同一)
    os.makedirs("data", exist_ok=True)
    for name, data in [("data/train.jsonl", train_samples), ("data/eval.jsonl", eval_samples)]:
        with open(name, "w") as f:
            for s in data:
                f.write(json.dumps({"text": format_sample(s), "category": s["category"]}, ensure_ascii=False) + "\n")
    for name, data in [("data/train_openai.jsonl", train_samples), ("data/eval_openai.jsonl", eval_samples)]:
        with open(name, "w") as f:
            for s in data:
                f.write(json.dumps(to_openai(s), ensure_ascii=False) + "\n")

    # 統計出力 (既存と同一形式)
    cats = Counter(s["category"] for s in samples)
    unique_outputs = len(set(s["output"] for s in samples))
    print(f"Generated {len(samples)} samples ({unique_outputs} unique outputs, {unique_outputs/len(samples)*100:.1f}% unique):")
    print(f"  Train: {len(train_samples)}, Eval: {n_eval}")
    print(f"\nBy category:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} ({count/len(samples)*100:.1f}%)")


if __name__ == "__main__":
    generate_dataset(60000)
```

**注意点**:
- `json.dumps` に `ensure_ascii=False` を追加する。カタカナを Unicode エスケープせず、そのまま出力するため。
- `random.seed(42)` をファイル冒頭に入れ、再現性を確保する。
- import 文: `json`, `random`, `os`, `collections.Counter`

### 2. `guppylm/eval_cases.py` — 全面書き換え

**現状**: 英語の eval ケース 16 件
**改修後**: 本仕様セクション 7 の EVAL_CASES をそのまま使用。`get_eval_cases()` 関数もそのまま維持。

### 3. 変更不要なファイル

以下は変更しない:

- `guppylm/config.py`
- `guppylm/model.py`
- `guppylm/dataset.py`
- `guppylm/prepare_data.py`
- `guppylm/train.py`
- `guppylm/inference.py`
- `guppylm/__init__.py`
- `guppylm/__main__.py`
