# PocketLLM
轻量级LLM
在线体验可以访问：https://huggingface.co/spaces/dw112/PocketLLM



## 项目简介

PocketLLM 是一个从零实现的轻量级语言模型与 Agent 系统，模型规模约 **0.2B参数**，可在单张 RTX 4090D 上 2 小时内完成全流程训练。项目覆盖预训练、SFT、RLHF/RLAIF（DPO/PPO/GRPO/CISPO）、Tool Use、Agentic RL、自适应思考与模型蒸馏等完整链路，所有核心算法使用**原生 PyTorch** 实现，不依赖 trl/peft 等高层封装。

---

## Agent 相关核心技术

### 1. 多轮 Agent RL 训练框架

基于 **GRPO / CISPO** 算法训练模型的工具调用能力。Agent 在推理时自主决策是否调用工具、选择哪个工具、如何组织参数，处理工具返回结果后可继续多轮交互直至完成目标。奖励信号在完整轨迹结束后延迟反馈，通过策略梯度优化模型的多步决策能力。

### 2. Tool-Use 系统

从零构建了完整的工具调用系统：

- 定义了 **8 类模拟工具**：数学计算、单位换算、天气查询、时间查询、汇率转换、翻译、文本长度统计、随机数生成
- 实现了工具调用的格式解析、参数校验与模拟执行引擎
- 通过 `tool_call` / `tool_response` 标签将工具交互注入对话模板
- 支持单轮中**并行调用多个工具**

### 3. Rollout Engine 抽象层

为 Agent RL 训练设计了可扩展的轨迹生成引擎：

- 支持 **PyTorch 原生推理**和 **SGLang RadixAttention** 两种后端
- 在训练中批量生成模型的多轮工具调用轨迹
- 计算 per-token log probabilities 用于策略梯度更新
- 抽象基类设计，便于扩展新的推理后端

### 4. OpenAI 兼容 API 服务

基于 **FastAPI** 搭建了兼容 OpenAI 协议的 API 服务，原生支持以下特性：

- `tool_calls` — 工具调用
- `reasoning_content` — 思考链内容展示
- `streaming` — 流式响应
- `open_thinking` — 自适应思考开关

可无缝接入 **FastGPT、Dify、Open-WebUI** 等第三方前端，实现 Agent 应用的端到端部署。

### 5. 完整强化学习训练链路

从数学公式出发，使用原生 PyTorch 实现了以下 RL 算法：

| 算法 | 说明 |
|------|------|
| **DPO** | 直接偏好优化，基于偏好对进行离线 RL |
| **PPO** | 在线策略优化，包含 Critic 网络 + GAE 优势估计 |
| **GRPO** | 组相对策略优化，去掉 Critic，通过组内比较提供信号 |
| **CISPO** | 裁剪重要性采样策略优化，GRPO 的 loss 变体 |

### 6. 自适应思考（Adaptive Thinking）

模型在推理时可通过 `open_thinking` 参数**动态开启/关闭** Chain-of-Thought 推理。思考过程以 `<think>` 标签包裹，与最终回答分离展示。该机制贯穿 SFT 数据构造、RL 奖励设计到 API 响应的全链路。

---

## 技术栈

`Python` `PyTorch` `Transformers` `FastAPI` `SGLang` `Streamlit` `DDP` `DeepSpeed`

## 推理引擎兼容

`Ollama` `llama.cpp (GGUF)` `vLLM` `SGLang` `MNN (端侧)`

---

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# CLI 推理
python eval_llm.py --load_from ./out/ --hidden_size 512 --num_hidden_layers 8

# 启动 OpenAI 兼容 API
python scripts/serve_openai_api.py --load_from ./out/ --port 8000

# 启动 Web Demo
streamlit run scripts/web_demo.py
```

---

*本项目基于 Apache 2.0 协议开源。*
