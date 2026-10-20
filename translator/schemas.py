from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

LANGUAGES = {
    "auto": "自动识别", "en": "英语", "ja": "日语", "ko": "韩语",
    "fr": "法语", "de": "德语", "es": "西班牙语", "ru": "俄语",
    "ar": "阿拉伯语", "pt": "葡萄牙语", "it": "意大利语", "vi": "越南语",
    "th": "泰语", "id": "印尼语", "tr": "土耳其语", "zh": "中文",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class TextRequest(StrictModel):
    text: str = Field(min_length=1, max_length=30000)
    language: str = "auto"
    domain: str = Field(default="general", min_length=1, max_length=80)

    @field_validator("language")
    @classmethod
    def language_valid(cls, value):
        if value not in LANGUAGES:
            raise ValueError("Unsupported language code")
        return value


class EntityInput(StrictModel):
    canonical_zh: str = Field(min_length=1, max_length=150)
    source_aliases: list[str] = Field(default_factory=list, max_length=40)
    zh_aliases: list[str] = Field(default_factory=list, max_length=40)
    language: str = "*"
    domain: str = Field(default="general", min_length=1, max_length=80)
    entity_type: str = Field(default="TERM", min_length=1, max_length=40)
    policy: Literal["normalize", "preserve"] = "normalize"
    enabled: bool = True

    @field_validator("source_aliases", "zh_aliases")
    @classmethod
    def valid_aliases(cls, values):
        if any(not v.strip() or len(v) > 150 for v in values):
            raise ValueError("Aliases must contain 1–150 characters")
        return list(dict.fromkeys(v.strip() for v in values))

    @field_validator("language")
    @classmethod
    def valid_language(cls, value):
        if value not in LANGUAGES.keys() | {"*"} or value == "auto":
            raise ValueError("Entity language must be a language code or *")
        return value


class EntityBatch(StrictModel):
    entities: list[EntityInput] = Field(min_length=1, max_length=2000)

