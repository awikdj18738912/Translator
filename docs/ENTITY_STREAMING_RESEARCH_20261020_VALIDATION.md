# 实体策略调研：验证记录

日期：2026-10-20。对应 [研究报告](ENTITY_STREAMING_RESEARCH_20261020.md)。只运行代码行为探针和既有测试；没有修改运行代码、真实实体库或重启服务，没有执行真实模型推理。

## 既有测试

在项目根目录执行：

```bash
PYTHONPATH=workbench python -m pytest \
  workbench/tests/test_regex_gate_integration.py \
  workbench/tests/test_window_refinement.py \
  workbench/tests/test_adaptive_translation.py \
  workbench/tests/test_token_previews.py \
  tests/test_terminology.py -q
```

实际解释器 `/home/aim0/anaconda3/bin/python`，pytest 7.4.4。结果：**55 passed, 8 warnings in 13.08s**。警告为 jieba / pkg_resources 及 namespace 弃用提示。

最初尝试工作台解释器 `/home/aim0/data/conda/envs/agentic-asr/bin/python -m pytest`，该环境没有 pytest，因此切换到已有测试环境；未安装或更改依赖。

## 精确匹配探针

以下可复现研究报告中的九条匹配观察。执行时使用工作台解释器和 `PYTHONPATH=workbench`。所有词条在内存构造；关闭通用规则保护以隔离字典实体逻辑。输出记录为行为观察，不代表期望行为。

```python
import json
from system.entity_store import EntityDefinition as E
from system.protection import EntityProtector

def entity(i, name, aliases=(), domain='trade', priority=0):
    return E(i, name, 'TERM', domain, 'normalize', priority, tuple(aliases))

def probe(label, entities, text):
    protector = EntityProtector(entities, enable_rule_protection=False)
    protected = protector.protect(text)
    result = protector.restore(protected.masked_text, protected)
    print(json.dumps({'case': label, 'input': text, 'output': result.text,
                      'entities': [s.entity_id for s in protected.spans]},
                     ensure_ascii=False))

probe('source_alias_survives', [entity(1, '最低起订量', ['MOQ'])], '请确认MOQ。')
probe('translation_unknown_alias',
      [entity(1, '最低起订量', ['minimum order quantity', 'MOQ'])], '请确认最小订单数。')
probe('ambiguous_alias_first',
      [entity(1, '甲公司', ['ABC']), entity(2, '乙公司', ['ABC'])], 'ABC交货。')
probe('ambiguous_alias_reversed',
      [entity(2, '乙公司', ['ABC']), entity(1, '甲公司', ['ABC'])], 'ABC交货。')
probe('general_before_specific',
      [entity(1, '通用含义', ['ABC'], domain='general'),
       entity(2, '外贸含义', ['ABC'])], 'ABC交货。')
probe('short_priority',
      [entity(1, '电池', ['锂电'], priority=9),
       entity(2, '锂离子电池', ['锂电池'])], '锂电池。')
probe('partial_prefix', [entity(1, '奥克米', ['ACME'])], 'ACME')
probe('later_boundary', [entity(1, '奥克米', ['ACME'])], 'ACMEX')
probe('short_entity', [entity(1, '苹果公司', ['苹果'])], '苹果汁。')
```

实际输出：

```jsonl
{"case":"source_alias_survives","input":"请确认MOQ。","output":"请确认最低起订量。","entities":[1]}
{"case":"translation_unknown_alias","input":"请确认最小订单数。","output":"请确认最小订单数。","entities":[]}
{"case":"ambiguous_alias_first","input":"ABC交货。","output":"甲公司交货。","entities":[1]}
{"case":"ambiguous_alias_reversed","input":"ABC交货。","output":"乙公司交货。","entities":[2]}
{"case":"general_before_specific","input":"ABC交货。","output":"通用含义交货。","entities":[1]}
{"case":"short_priority","input":"锂电池。","output":"电池池。","entities":[1]}
{"case":"partial_prefix","input":"ACME","output":"奥克米","entities":[1]}
{"case":"later_boundary","input":"ACMEX","output":"ACMEX","entities":[]}
{"case":"short_entity","input":"苹果汁。","output":"苹果公司汁。","entities":[1]}
```

## 精修 session 探针

使用工作台解释器执行以下代码。Refiner 被 mock 替换为原样返回的类；临时库在退出时删除。`create_app` 的“Loading AgenticASR Refiner...”是既有日志，探针没有加载模型权重。未注入原生数字规则库，该探针用于隔离实体规范化与 regex 门控，不是全部生产组件集成验证；所用输入也没有数字规则需求。

```python
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from system import web_app
from system.entity_store import EntityStore

class Refiner:
    calls = []
    def __init__(self, *args, **kwargs):
        pass
    def refine(self, text, **kwargs):
        self.calls.append(text)
        return text, 1.0

with TemporaryDirectory() as tmp:
    db = Path(tmp) / 'entities.db'
    EntityStore(db).upsert_entity('最低起订量', aliases=('MOQ',), domain='trade',
                                  normalization_policy='normalize')
    with patch.object(web_app, 'TransformersRefiner', Refiner):
        app = web_app.create_app(Path('unused'), 'cpu', 'http://asr.invalid',
            None, 128, None, entity_db=db, entity_fuzzy_mode='off',
            refinement_gate_mode='regex')
        session = app.state.new_text_refinement_session('trade')
        for revision, text, final in [
            (1, '请确认MOQ。', False),
            (2, '请确认MOQ。请按时发货。请附装箱单。', True),
        ]:
            before = len(Refiner.calls)
            r = session.update(text, revision=revision, final=final)
            print(json.dumps({'revision': revision, 'final': final, 'input': text,
                'output': r['clean_text'], 'new_model_calls': len(Refiner.calls)-before,
                'gate': [x['action'] for x in r['refinement_gate_decisions']]},
                ensure_ascii=False))
```

实际输出：

```jsonl
{"revision":1,"final":false,"input":"请确认MOQ。","output":"请确认最低起订量。","new_model_calls":0,"gate":["defer"]}
{"revision":2,"final":true,"input":"请确认MOQ。请按时发货。请附装箱单。","output":"请确认最低起订量。请按时发货。请附装箱单。","new_model_calls":1,"gate":["refine"]}
```

后续实施需新增相应回归覆盖，尤其是冲突顺序、重叠词、短中文别名、跨 chunk 边界、翻译修订失效，以及已解决实体不单独触发模型。当前既有测试通过不能证明这些未覆盖情形正确。
