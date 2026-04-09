"""Held-out conversation evaluation pack for Guppy (katakana version).

Hand-authored test cases. Each has a user message and expected traits
in Guppy's response (keywords, tone, character consistency).
"""


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


def get_eval_cases():
    return list(EVAL_CASES)
