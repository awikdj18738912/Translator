# LMT 接入 Codex-web：离线文本流式回放

本入口使用真实 GPU 模型，把外语文本的不断增长/修正快照交给 LMT，再把最新完整中文译文交给 Codex-web 的原精修回调。它不运行音频识别，不是浏览器音频 WebSocket 集成。

```text
外语文本快照（可回改）
  → LMT-60-0.6B / CTranslate2 GPU INT8+FP16 / beam=3
  → 最新完整中文译文快照（替换上一版）
  → Codex-web fixed_k3_context_v2 分窗与相邻只读上下文
  → Codex-web 实体保护 / 原始 GPU Refiner / 补丁解析与守卫
  → 接受精修候选或回退到精修前的中文
```

## 本机启动

在 `/home/aim0/桌面/tan/ASR/Translator` 执行：

```bash
bash scripts/run_codexweb_translation.sh \
  --gpu 1 --interval 0.5 \
  --output results/my_translation_replay
```

结果目录不能已有 `events.jsonl`，再次运行请更换 `--output`。无需启动 ASR 或原来的 Web 服务。进程完成后退出并释放模型显存。

本机脚本默认使用 `/home/aim0/data/conda/envs/agentic-asr/bin/python`（Transformers 5）和本项目 `.runtime/lmt-deps` 中的 CTranslate2 4.8.3。Refiner 当前权重的 TokenizersBackend 类型需要新版 Transformers；仅用旧评测环境的 4.57.6 无法加载。`REPLAY_PYTHON` 可以覆盖 Python 路径。路径均可通过 CLI 参数更换，完整参数见 `--help`。

默认回放八语种报价、数量改口、ASR式源文回改、iPhone实体样例和多句窗口，共12个场景。模型每次只看当前外语片段，不把整场会话无界追加到上下文。回放按输入间隔调度，单次处理未完成会顺序等待，不建立无界任务队列；因此不能将该实验的进程墙时解释为实时语音延迟。

## 自定义输入

`--cases path.json --case-ids sample_id` 可以指定样例。数据格式：

```json
{"cases":[{"id":"sample_id","language":"en","chunks":["We need 100 iPhone units",", with delivery by Friday."],"reference":"我们需要100部iPhone，周五前交货。"}]}
```

`chunks` 会逐块拼接成源文；也可以用 `prefixes` 给出每次完整源文快照，允许 ASR 类回改。两者任选其一。当前源语代码：en/fr/de/es/ru/ja/ko/ar。中文目标固定，revision 从1递增，终态后不能继续更新该会话。

## 实体库与精修

默认创建本次独立演示库，含 iPhone、SUS304、AcmeFlow、阀门、不锈钢。不会把“泵”等含义不同的词设为阀门别名。

可传 `--entity-db /path/to/Codex-web/data/entities.db`，脚本会用 SQLite 只读连接备份到本次目录，模型使用副本。不会给用户原库增删词条。当前领域为 trade，同时包含 general。实体库作用在中文精修前后，尚未加入翻译前的跨语术语约束。

联调显式关闭 ASR 置信度门控（没有真实 ASR 置信度），以验证所有就绪窗口都能调用精修。实体模糊匹配关闭，仅使用已登记的精确规则。原窗口逻辑会等够K=3及后续上下文，短句可能直到final才精修；待处理中文会立即显示。终态拒绝窗口重跑配置与原Web的 final_refinement_mode 一致。

这次复用的是原 `create_app.refine_update`、固定分窗、实体与数字守卫；没有模拟音频复核、CTC、WebSocket队列或浏览器UI。Refiner只看中文，因此接受候选不等于外语信息完整/正确。

## 产物

- `events.jsonl`：每次外语、翻译、精修结果、revision、时延、守卫和窗口状态。
- `finals.json`：每个场景最终结果。
- `refiner_traces.json`：真实精修调用输入、候选、结构化补丁及回退痕迹；也包含不调用模型的pending窗口。
- `metadata.json`：运行环境、模型路径、词库快照、代码SHA256、GPU占用及计时口径。
- `REPORT.md`：便于阅读的最终对照表。

本次新增 `Codex-web/system/offline_text_refinement.py`，并在 `web_app.py` 增加 Python 工厂 `app.state.new_text_refinement_session(domain)`。原HTTP/音频WebSocket接口未添加翻译步骤；后续浏览器集成需要再接对应输入与输出事件。
