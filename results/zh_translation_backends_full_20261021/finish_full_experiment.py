#!/usr/bin/env python3
"""Continue full scoring and reporting when all generation groups finish."""
import json, subprocess, sys, time
from pathlib import Path
base=Path(__file__).resolve().parent
while not (base/'generation_complete.json').exists():
    time.sleep(10)
print('All six generation groups complete; starting full scoring.',flush=True)
subprocess.run(['/home/aim0/data/venvs/translation-bench/bin/python',str(base/'run_metric_pipeline.py'),'--stage','all'],check=True,cwd=base.parents[1])
doc=base.parents[1]/'docs/INDEX_TRANSLATE_2B_QUANTIZATION_COMPARISON_20261021.md'
addition='\n\n## BF16、FP8、NVFP4 在 vLLM 与 Transformers 上的全量对照\n\n六组均覆盖八种语言、每语种 1,012 条，五项质量指标及翻译吞吐、分语种结果、宏平均和置信区间见 [完整后端对照报告](../results/zh_translation_backends_full_20261021/BACKEND_COMPARISON.md)。\n'
if '完整后端对照报告' not in doc.read_text():
    with doc.open('a') as stream:stream.write(addition)
(base/'experiment_complete.json').write_text(json.dumps({'status':'complete','time':time.time()}))
print('EXPERIMENT COMPLETE',flush=True)
