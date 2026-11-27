"""引数と YAML で共有する設定。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_FIELDS = (
    "min_topic_size",
    "nr_topics",
    "n_neighbors",
    "reduce_outliers",
    "embedding_model",
    "exclude",
    "protect",
    "fillers",
    "pos",
    "stopwords",
)

def _default_pos() -> list[str]:
    from bertopic_ja._tokenizer import DEFAULT_POS

    return list(DEFAULT_POS)


@dataclass
class Config:
    """フィット時の設定。引数と YAML は同じ項目を持つ。"""

    min_topic_size: int = 10
    nr_topics: int | str | None = None
    n_neighbors: int | None = None
    reduce_outliers: bool = True
    embedding_model: Any = None
    exclude: list[str] = field(default_factory=list)
    protect: list[str] = field(default_factory=list)
    fillers: list[str] = field(default_factory=list)
    pos: list[str] = field(default_factory=_default_pos)
    stopwords: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None, overrides: dict[str, Any] | None = None) -> Config:
        merged: dict[str, Any] = dict(data or {})
        if overrides:
            merged.update(overrides)
        unknown = sorted(set(merged) - set(_FIELDS))
        if unknown:
            names = ", ".join(unknown)
            raise ValueError(f"未知の設定項目です: {names}")
        for key in ("exclude", "protect", "fillers", "stopwords"):
            value = merged.get(key)
            if value is None:
                merged[key] = []
            elif not isinstance(value, list):
                raise ValueError(f"{key} は文字列のリストです。")
            else:
                merged[key] = [str(item) for item in value]
        if "pos" not in merged or merged["pos"] is None:
            merged.pop("pos", None)
        elif not isinstance(merged["pos"], list):
            raise ValueError("pos は文字列のリストです。")
        else:
            merged["pos"] = [str(item) for item in merged["pos"]]
        if "min_topic_size" in merged:
            merged["min_topic_size"] = int(merged["min_topic_size"])
        if "n_neighbors" in merged and merged["n_neighbors"] is not None:
            merged["n_neighbors"] = int(merged["n_neighbors"])
        return cls(**merged)

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ValueError("YAML の先頭は設定項目のマップにしてください。")
        return cls.from_mapping(loaded, overrides)

    def to_dict(self) -> dict[str, Any]:
        embedding_model = self.embedding_model if isinstance(self.embedding_model, str) else None
        return {
            "min_topic_size": self.min_topic_size,
            "nr_topics": self.nr_topics,
            "n_neighbors": self.n_neighbors,
            "reduce_outliers": self.reduce_outliers,
            "embedding_model": embedding_model,
            "exclude": list(self.exclude),
            "protect": list(self.protect),
            "fillers": list(self.fillers),
            "pos": list(self.pos),
            "stopwords": list(self.stopwords),
        }

    def to_yaml(self, path: str | Path) -> None:
        text = yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False)
        Path(path).write_text(text, encoding="utf-8")
