import sqlite3

from translator.import_codex import read_entities


def test_codex_import_is_readonly_and_separates_explicit_aliases(tmp_path):
    source = tmp_path / "entities.db"
    with sqlite3.connect(source) as db:
        db.executescript("""
          CREATE TABLE protected_entities(id INTEGER,canonical_text TEXT,entity_type TEXT,domain TEXT,normalization_policy TEXT,enabled INTEGER);
          CREATE TABLE entity_aliases(entity_id INTEGER,alias TEXT);
          INSERT INTO protected_entities VALUES(1,'最低起订量','TERM','trade','normalize',1);
          INSERT INTO entity_aliases VALUES(1,'MOQ');
          INSERT INTO entity_aliases VALUES(1,'最小订单量');
        """)
    before = source.read_bytes()
    entities = read_entities(source)
    assert source.read_bytes() == before
    assert entities[0].canonical_zh == "最低起订量"
    assert entities[0].source_aliases == ["MOQ"]
    assert entities[0].zh_aliases == ["最小订单量"]
    assert entities[0].domain == "trade"

