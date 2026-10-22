# 中文语音精修与外语翻译工作台

这是独立的 `trade-translator` 项目。当前工作台流程为 **中文 ASR → 中文精修 → 所选外语翻译**，三列分别流式展示对应结果。支持英语、法语、德语、西班牙语、俄语、日语、韩语和阿拉伯语作为翻译目标。

中文精修保留本地数字规则、实体保护、正则门控与结果校验；翻译只接收已经完成精修的中文前缀。网页默认端口为 `8083`，ASR 为 `8766`。

## 项目结构

```text
Translator/
├── pyproject.toml    项目元数据与基础应用打包配置
├── translator/       早期基础翻译应用 Python 包（保留原接口）
├── workbench/        本地 GPU 流式工作台及其网页前端
├── zh-itn/           中文数字 ITN 规则源码与原生库
├── tests/            基础应用测试
├── scripts/          模型准备、评测与维护脚本
└── docs/             设计及运行说明
```

数字规则库位于项目根目录 `zh-itn/zh-itn/`，供 `workbench/` 工作台调用。工作台代码与静态前端在 `workbench/` 内维护；它从根目录 `zh-itn/` 加载本地规则库。

工作台的模型配置、启动命令、端口和运行记录见 [`workbench/README.md`](workbench/README.md)。基础应用开发和打包配置见 `pyproject.toml`。
