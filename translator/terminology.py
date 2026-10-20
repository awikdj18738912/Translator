"""Exact, domain-scoped terminology; no speculative phonetic replacements."""
from collections import Counter
from dataclasses import dataclass
import re
from uuid import uuid4

TOKEN_RE = re.compile(r"__TR_[a-f0-9]{12}_\d{4}__")


@dataclass
class ProtectedText:
    text: str
    replacements: dict[str, str]
    matches: list[dict]
    warnings: list[str]

    def restore(self, candidate: str, *, ordered=False):
        tokens = TOKEN_RE.findall(candidate)
        if Counter(tokens) != Counter(self.replacements.keys()):
            raise ValueError("受保护实体或数字占位符丢失、增加或重复")
        if ordered and tokens != list(self.replacements):
            raise ValueError("精修改变了受保护信息的顺序")
        for token, target in self.replacements.items():
            candidate = candidate.replace(token, target)
        if "__TR_" in candidate:
            raise ValueError("存在未恢复的占位符")
        return candidate


def exact_pattern(alias):
    left = r"(?<![A-Za-z0-9_])" if alias[0].isascii() and alias[0].isalnum() else ""
    right = r"(?![A-Za-z0-9_])" if alias[-1].isascii() and alias[-1].isalnum() else ""
    return re.compile(left + re.escape(alias) + right, flags=re.IGNORECASE)


def protect(text, entities, *, domain, language, phase, facts=False):
    prefix = uuid4().hex[:12]
    candidates, warnings = [], []
    for entity in entities:
        if not entity["enabled"] or entity["domain"] not in {"general", domain}:
            continue
        if phase == "source" and language != "auto" and entity["language"] not in {"*", language}:
            continue
        aliases = entity["source_aliases"] if phase == "source" else [entity["canonical_zh"], *entity["zh_aliases"]]
        for alias in aliases:
            for match in exact_pattern(alias).finditer(text):
                observed = match.group()
                target = observed if entity["policy"] == "preserve" else entity["canonical_zh"]
                candidates.append({"start": match.start(), "end": match.end(), "observed": observed,
                                   "target": target, "entity_id": entity["id"], "domain": entity["domain"]})
    # Prefer the longest phrase; a domain-specific definition overrides general at the same span.
    by_span = {}
    for c in candidates:
        by_span.setdefault((c["start"], c["end"]), []).append(c)
    safe = []
    for span, options in by_span.items():
        specific = [c for c in options if c["domain"] == domain]
        options = specific or options
        if len({c["target"] for c in options}) != 1:
            warnings.append(f"术语有多个标准译名，未自动替换：{text[span[0]:span[1]]}")
            continue
        safe.append(options[0])
    selected = []
    for c in sorted(safe, key=lambda c: (-(c["end"] - c["start"]), c["start"])):
        if not any(c["start"] < old["end"] and old["start"] < c["end"] for old in selected):
            selected.append(c)
    if facts:
        # Keep quantity-unit expressions intact during Chinese-only refinement.
        fact_pattern = r"(?:[A-Z]{2,}[A-Z0-9_-]*|[$€£¥￥]?\d+(?:[.,:/-]\d+)*(?:\s*(?:%|％|元|美元|欧元|人民币|万元|万|千|亿|个|件|台|箱|吨|公斤|千克|克|毫米|厘米|米|天|日|周|月|年|小时|分钟|kg|mm|cm|pcs|USD|EUR|CNY|m))?)"
        for match in re.finditer(fact_pattern, text):
            if not any(match.start() < c["end"] and c["start"] < match.end() for c in selected):
                selected.append({"start": match.start(), "end": match.end(), "observed": match.group(),
                                 "target": match.group(), "entity_id": None, "domain": domain})
    parts, replacements, cursor = [], {}, 0
    selected.sort(key=lambda c: c["start"])
    for i, c in enumerate(selected):
        token = f"__TR_{prefix}_{i:04d}__"
        parts.extend((text[cursor:c["start"]], token))
        replacements[token] = c["target"]
        cursor = c["end"]
    parts.append(text[cursor:])
    return ProtectedText("".join(parts), replacements, selected, list(dict.fromkeys(warnings)))

