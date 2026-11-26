"""Sudachi によるトピック語の抽出。

フィラーはトピック語から外す。本文からの削除は fillers を渡したときだけで、
そのときもトークン単位で行い、名詞・連体詞は残す。
"""

from __future__ import annotations

from functools import lru_cache

from sklearn.feature_extraction.text import CountVectorizer

from bertopic_ja._config import Config

# 感動詞として取りこぼしたときのための完全一致。指示語の「あの」は含めない。
FILLER_TOKENS = frozenset(
    {
        "えー",
        "えーっと",
        "えーと",
        "あのー",
        "あのぉ",
        "まあ",
        "まぁ",
        "まー",
    }
)

# 本文からフィラーを消すとき、この品詞は残す。「あの本」の「あの」は連体詞。
_KEEP_WHEN_STRIPPING = frozenset({"名詞", "動詞", "形容詞", "連体詞"})


@lru_cache(maxsize=1)
def tokenizer():
    """Sudachi のタグガーと分割モード C を返す。"""
    try:
        from sudachipy import Dictionary, SplitMode
    except ImportError:  # sudachipy 0.6
        from sudachipy.dictionary import Dictionary
        from sudachipy.tokenizer import Tokenizer

        SplitMode = Tokenizer.SplitMode

    dictionary = Dictionary()
    tagger = dictionary.create() if hasattr(dictionary, "create") else dictionary.tokenizer()
    return tagger, SplitMode.C


def _forms(morpheme) -> tuple[str, str]:
    surface = morpheme.surface()
    try:
        normalized = morpheme.normalized_form()
    except Exception:
        normalized = surface
    if not normalized:
        normalized = surface
    return surface, normalized


def iter_morphemes(text: str):
    if not text or not str(text).strip():
        return
    tagger, mode = tokenizer()
    for morpheme in tagger.tokenize(str(text), mode):
        yield morpheme


def strip_fillers(text: str, fillers: list[str]) -> str:
    """指定したフィラーを、トークン単位で本文から外す。"""
    targets = set(fillers)
    if not targets or not text:
        return "" if text is None else str(text)
    kept: list[str] = []
    for morpheme in iter_morphemes(text):
        surface, normalized = _forms(morpheme)
        pos = morpheme.part_of_speech()[0]
        listed = surface in targets or normalized in targets
        if listed and pos not in _KEEP_WHEN_STRIPPING:
            continue
        kept.append(surface)
    return "".join(kept)


def explain_text(text: str, config: Config) -> dict[str, list]:
    """1文について、残す語と落とした理由を返す。"""
    kept: list[str] = []
    dropped: list[dict[str, str]] = []
    protect = set(config.protect)
    exclude = set(config.exclude)
    for morpheme in iter_morphemes(text):
        surface, normalized = _forms(morpheme)
        pos = morpheme.part_of_speech()
        pos_main = pos[0] if pos else ""
        pos_sub = pos[1] if len(pos) > 1 else ""
        token = normalized or surface
        reason = _drop_reason(token, surface, pos_main, pos_sub, protect, exclude)
        if reason is None:
            kept.append(token)
        else:
            dropped.append({"token": surface, "reason": reason})
    return {"kept": kept, "dropped": dropped}


def _drop_reason(
    token: str,
    surface: str,
    pos_main: str,
    pos_sub: str,
    protect: set[str],
    exclude: set[str],
) -> str | None:
    if token in protect or surface in protect:
        return None
    if token in FILLER_TOKENS or surface in FILLER_TOKENS:
        return "フィラー"
    if pos_main == "感動詞":
        return "感動詞"
    if token in exclude or surface in exclude:
        return "除外"
    if pos_main != "名詞":
        return pos_main or "品詞"
    if pos_sub in {"代名詞", "数詞"}:
        return pos_sub
    if len(token) <= 1:
        return "1文字"
    if token.isdigit():
        return "数"
    return None


def build_vectorizer(config: Config) -> CountVectorizer:
    """トピック語専用の CountVectorizer。クラスタ分けには使わない。"""
    protect = set(config.protect)
    exclude = set(config.exclude)

    def analyze(text: str) -> list[str]:
        words: list[str] = []
        for morpheme in iter_morphemes(text):
            _surface, normalized = _forms(morpheme)
            pos = morpheme.part_of_speech()
            pos_main = pos[0] if pos else ""
            pos_sub = pos[1] if len(pos) > 1 else ""
            token = normalized or _surface
            if _drop_reason(token, _surface, pos_main, pos_sub, protect, exclude) is None:
                words.append(token)
        return words

    return CountVectorizer(analyzer=analyze, token_pattern=None)
