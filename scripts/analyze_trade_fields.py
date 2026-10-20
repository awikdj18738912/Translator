"""Read existing translation case outputs and diagnose quoted amount/valve terms.

This is a narrow diagnostic over eight parallel, constructed order examples per
model. It does not estimate production accuracy.
"""
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "ct2_int8_float16_beam3": ROOT / "results/translation_streaming_20261017/beam3_realtime",
    "transformers_fp16_beam3": ROOT / "results/translation_streaming_20261017/fp16_beam3",
}
OUTPUT_DIR = ROOT / "results/multilingual_mt_accuracy_20261017"
NUMBER = r"(?<![\d])(?:\d{1,3}(?:,\d{3})+|\d+)(?:[.,]\d+)?(?![\d])"
CURRENCY = r"(?:US\s*\$|USD|美元|美金|美圆|\$)"
AMOUNT_THEN_CURRENCY = re.compile(rf"(?P<amount>{NUMBER})\s*(?P<currency>{CURRENCY})", re.I)
CURRENCY_THEN_AMOUNT = re.compile(rf"(?P<currency>{CURRENCY})\s*(?P<amount>{NUMBER})", re.I)
CURRENCY_RE = re.compile(CURRENCY, re.I)
CHINESE_NUMERALS = re.compile(r"[零〇一二两三四五六七八九十百千万亿点０-９]+")


def normalize_amount(raw: str):
    """Normalize ordinary decimal comma/point and grouped integers safely."""
    value = raw
    if "." in value and "," in value:
        # Common English grouping plus decimal form, e.g. 1,234.50.
        value = value.replace(",", "")
    elif "," in value:
        chunks = value.split(",")
        if len(chunks) > 2:
            if all(len(chunk) == 3 for chunk in chunks[1:]):
                value = "".join(chunks)
            else:
                return None, "ambiguous_comma_grouping"
        elif len(chunks[1]) == 3 and len(chunks[0]) <= 3:
            # Could mean a thousands separator or a three-decimal amount.
            return None, "ambiguous_comma_grouping"
        else:
            value = ".".join(chunks)
    try:
        return format(Decimal(value).normalize(), "f"), None
    except InvalidOperation:
        return None, "unparseable_number"


def extract_amount(text: str):
    matches = []
    for pattern in (AMOUNT_THEN_CURRENCY, CURRENCY_THEN_AMOUNT):
        for match in pattern.finditer(text):
            amount = match.group("amount")
            normalized, issue = normalize_amount(amount)
            matches.append({
                "surface_amount": amount,
                "currency_surface": match.group("currency"),
                "normalized_amount": normalized,
                "span": [match.start(), match.end()],
                "issue": issue,
            })
    # The two patterns should not overlap, but de-duplicate defensively.
    unique = {tuple(item["span"]): item for item in matches}
    matches = list(unique.values())
    if matches:
        status = "explicit_arabic_amount_and_currency"
        if len(matches) != 1 or matches[0]["issue"]:
            status = "uncertain_amount_structure"
    elif CURRENCY_RE.search(text) and CHINESE_NUMERALS.search(text):
        status = "uncertain_chinese_numeral_amount"
    elif CURRENCY_RE.search(text):
        status = "uncertain_currency_without_arabic_amount"
    else:
        status = "no_explicit_amount_and_currency"
    return {"status": status, "matches": matches}


def load_orders(path: Path):
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("mode") != "retranslate":
                continue
            case_id = row.get("case_id", "")
            if not case_id.endswith("_order"):
                continue
            rows.append((line_number, row))
    rows.sort(key=lambda item: item[1]["case_id"])
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    models = {}
    for model, directory in INPUTS.items():
        # Each directory contains the model-specific files for both models.
        models[model] = {}
        model_results = {}
        for model_file in ("m2m100_418M", "small100"):
            path = directory / f"{model_file}.cases.jsonl"
            rows = load_orders(path)
            details = []
            for line_number, row in rows:
                translation = row.get("final_translation", "")
                amount = extract_amount(translation)
                definite_preserved = (
                    amount["status"] == "explicit_arabic_amount_and_currency"
                    and len(amount["matches"]) == 1
                    and amount["matches"][0]["normalized_amount"] == "12.5"
                )
                details.append({
                    "case_id": row["case_id"],
                    "language": row.get("language"),
                    "output": translation,
                    "amount_extraction": amount,
                    "explicit_quote_amount_currency_is_12_50": definite_preserved,
                    "contains_simplified_valve_character_阀": "阀" in translation,
                    "contains_traditional_valve_character_閥": "閥" in translation,
                    "source_reference_amount": "12.50 USD (explicit in source/reference template)",
                    "source_jsonl_line": line_number,
                })
            models[model][model_file] = {
                "input_file": str(path.relative_to(ROOT)),
                "case_count": len(details),
                "languages": [item["language"] for item in details],
                "explicit_12_50_with_currency_count": sum(item["explicit_quote_amount_currency_is_12_50"] for item in details),
                "explicit_12_50_with_currency_denominator": len(details),
                "valve_term_阀_count": sum(item["contains_simplified_valve_character_阀"] for item in details),
                "valve_term_閥_count_diagnostic_only": sum(item["contains_traditional_valve_character_閥"] for item in details),
                "uncertain_amount_count": sum(item["amount_extraction"]["status"].startswith("uncertain") for item in details),
                "rows": details,
            }
    report = {
        "title": "Trade field diagnostics: quoted USD amount and valve term",
        "scope": "Only mode=retranslate order cases from two existing beam=3 result directories; no model inference was run.",
        "sampling_note": "Each model has eight parallel-language translations of the same constructed quote template. These are not random business samples and cannot estimate production accuracy.",
        "amount_rule": "Count as retained only one unambiguous Arabic-numeral amount explicitly adjacent to a USD/美元/美金 currency marker and normalized to Decimal 12.5 (12.50). Chinese-numeral or unclear structures are marked uncertain, not automatically scored wrong.",
        "valve_rule": "Count literal simplified Chinese character 阀 only. This is a term-hit diagnostic, not sentence or translation accuracy; traditional 閥 is reported separately for visibility.",
        "models": models,
    }
    json_path = OUTPUT_DIR / "trade_field_diagnostics.json"
    md_path = OUTPUT_DIR / "trade_field_diagnostics.trade.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Trade field diagnostics",
        "",
        "This reads existing `mode=retranslate` beam=3 results only; no model inference was run. Each model has eight parallel-language translations of the same constructed quote template. These are not random business samples and cannot estimate production accuracy.",
        "",
        "An amount counts as retained only when one unambiguous Arabic-numeral amount is explicitly adjacent to a USD/美元/美金 marker and normalizes to 12.50. Chinese-numeral or unclear structures are marked uncertain rather than automatically scored wrong. The `阀` count is a literal term-hit diagnostic, not sentence or translation accuracy; traditional `閥` is shown separately.",
        "",
        "| Result set | Model | Explicit 12.50 + currency | Literal 阀 | Traditional 閥 (diagnostic) | Uncertain amount rows |",
        "|---|---|---:|---:|---:|---:|",
    ]
    labels = {"ct2_int8_float16_beam3": "CT2 INT8/FP16 beam=3", "transformers_fp16_beam3": "Transformers FP16 beam=3"}
    for result_set, set_data in models.items():
        for model_file, data in set_data.items():
            lines.append(
                f"| {labels[result_set]} | {model_file} | {data['explicit_12_50_with_currency_count']}/{data['explicit_12_50_with_currency_denominator']} | {data['valve_term_阀_count']}/{data['case_count']} | {data['valve_term_閥_count_diagnostic_only']}/{data['case_count']} | {data['uncertain_amount_count']} |"
            )
    lines.extend(["", "## Per-case outputs", ""])
    for result_set, set_data in models.items():
        for model_file, data in set_data.items():
            lines.extend([f"### {labels[result_set]} — {model_file}", "", "| Language | Output | Extracted amount + currency | 12.50 retained | 阀 |", "|---|---|---|---|---:|"])
            for item in data["rows"]:
                found = item["amount_extraction"]["matches"]
                extraction = "; ".join(
                    f"{m['surface_amount']} {m['currency_surface']} → {m['normalized_amount'] or m['issue']}"
                    for m in found
                ) or item["amount_extraction"]["status"]
                output = item["output"].replace("|", "\\|")
                lines.append(f"| {item['language']} | {output} | {extraction} | {'yes' if item['explicit_quote_amount_currency_is_12_50'] else 'no/uncertain'} | {'yes' if item['contains_simplified_valve_character_阀'] else 'no'} |")
            lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "summary": {
        name: {model: {"amount": f"{data['explicit_12_50_with_currency_count']}/{data['explicit_12_50_with_currency_denominator']}", "valve": f"{data['valve_term_阀_count']}/{data['case_count']}", "uncertain": data['uncertain_amount_count']} for model, data in group.items()}
        for name, group in models.items()
    }}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
