"""One-way, read-only import from a Codex-web entity database."""
import argparse
from pathlib import Path
import re
import sqlite3

from .schemas import EntityInput
from .store import Store


def read_entities(source: Path):
    source = source.resolve(strict=True)
    db = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        rows = db.execute("SELECT id,canonical_text,entity_type,domain,normalization_policy,enabled FROM protected_entities").fetchall()
        aliases = {}
        for row in db.execute("SELECT entity_id,alias FROM entity_aliases"):
            aliases.setdefault(row["entity_id"], []).append(row["alias"])
        entities = []
        for row in rows:
            values = aliases.get(row["id"], [])
            entities.append(EntityInput(
                canonical_zh=row["canonical_text"], entity_type=row["entity_type"], domain=row["domain"],
                source_aliases=[v for v in values if not re.search(r"[\u3400-\u9fff]", v)],
                zh_aliases=[v for v in values if re.search(r"[\u3400-\u9fff]", v)],
                language="*", policy=row["normalization_policy"], enabled=bool(row["enabled"])))
        return entities
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="从 Codex-web 只读导入实体，不修改来源库")
    parser.add_argument("source", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    entities = read_entities(args.source)
    if args.dry_run:
        print(f"可导入 {len(entities)} 个实体；未写入数据库。请检查源语言别名与领域是否适用于翻译场景。")
    else:
        Store(args.data_dir / "translator.sqlite3").upsert_entities(entities)
        print(f"已导入 {len(entities)} 个实体；来源库未修改。请补充外语别名，混合语言别名需人工核对。")


if __name__ == "__main__":
    main()

