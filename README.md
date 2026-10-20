# 多语言语音翻译工作台

这是独立的 `trade-translator` 项目。它包含基础翻译应用，以及用于本地 GPU 流式转录、翻译和精修的工作台。

## 项目结构

```text
Translator/
├── pyproject.toml    项目元数据与基础应用打包配置
├── translator/       基础翻译应用 Python 包
├── workbench/        本地 GPU 流式工作台及其网页前端
├── zh-itn/           中文数字 ITN 规则源码与原生库
├── tests/            基础应用测试
├── scripts/          模型准备、评测与维护脚本
└── docs/             设计及运行说明
```

数字规则库位于项目根目录 `zh-itn/zh-itn/`，供 `workbench/` 工作台调用。工作台代码与静态前端在 `workbench/` 内维护；它从根目录 `zh-itn/` 加载本地规则库。

工作台的模型配置、启动命令、端口和运行记录见 [`workbench/README.md`](workbench/README.md)。基础应用开发和打包配置见 `pyproject.toml`。
