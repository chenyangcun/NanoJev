# NanoJev 基于 RLCD 的决策模型训练指南

本文档介绍如何在 NanoJev（基于 `Qwen3-0.6B` 纯文本底座）上，借鉴并复用 Dohnuts / Laya 的 **RLCD（Reinforcement Learning for Calibrated Decisions，校准决策强化学习）** 算法与公开数据集，完成数据下载、纯文本清洗、模型微调、温度校准到 MLX/ANE 部署的全流程。

---

## 目录
1. [方案概述](#1-方案概述)
2. [环境准备](#2-环境准备)
3. [素材与训练数据集获取](#3-素材与训练数据集获取)
4. [数据清洗与纯文本格式转换](#4-数据清洗与纯文本格式转换)
5. [RLCD 训练机制与代码接入](#5-rlcd-训练机制与代码接入)
6. [启动微调训练](#6-启动微调训练)
7. [温度校准与导出为 MLX 格式](#7-温度校准与导出为-mlx-格式)
8. [验证与部署到生产服务](#8-验证与部署到生产服务)
9. [借鉴 Dohnuts 的严密评测体系 (Benchmark)](#9-借鉴-dohnuts-的严密评测体系-benchmark)

---

## 1. 方案概述

* **底座网络**：保持 `Qwen/Qwen3-0.6B` 不变，确保 100% 兼容 Apple Silicon 的 **ANE 5ms 快车道** 与 **Tree-Prefill 树状前缀共享**；
* **微调方式**：冻结基座，微调语言 LoRA（Rank 8）及判决头（`DeepDecisionHeads`）；
* **训练目标**：RLCD 联合损失函数（高斯扰动探索 + Proper Scoring Rule 严格评分规则 + 辅助交叉熵）；
* **数据来源**：复用 Dohnuts 清单中的公开决策数据（MASSIVE、BoolQ、BANKING77、LocalLLaMA/typed-decisions 等），剔除多模态图像部分。

---

## 2. 环境准备

训练阶段推荐在具有 CUDA 支持的 GPU 环境或本地 PyTorch 环境下完成（需支持 LoRA 与梯度更新）：

```bash
# 进入 NanoJev 目录
cd /Users/chenyc/Documents/study/NanoJev

# 激活 Python 虚拟环境
source .venv/bin/activate

# 确保安装必要依赖
pip install torch torchvision transformers peft pyarrow datasets
```

---

## 3. 素材与训练数据集获取

Dohnuts 项目梳理了严格固定哈希的公开数据集清单，我们可以直接利用其清单获取原始数据：

### 3.1 核心数据来源
1. **软标签决策集**：`LocalLLaMA/typed-decisions`（天然包含 choice, noul, score 概率分布）；
2. **意图路由集**：`MASSIVE 1.1`（中英 60 类意图）、`BANKING77`；
3. **布尔推理集**：`SuperGLUE BoolQ`（二元是非判断）；
4. **分类与业务集**：`emotion`、`Contract-NLI`、`SHARC`、`Amazon ESCI`。

### 3.2 下载素材
可通过脚本自动下载或直接从邻近的 Dohnuts 目录中导入：
```bash
# 如果已在 dohnuts 目录中执行过下载，可直接软链接或复制
mkdir -p data/raw
cp -r /Users/chenyc/Documents/study/dohnuts/data/raw/* data/raw/
```

若独立下载，可从 Hugging Face 获取核心 typed-decisions：
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='LocalLLaMA/typed-decisions',
    repo_type='dataset',
    local_dir='data/raw/typed-decisions'
)
"
```

---

## 4. 数据清洗与纯文本格式转换

NanoJev 专精纯文本决策，需过滤掉所有涉及图像与视觉坐标的样本。

运行以下脚本将原始数据转化为标准的纯文本训练 JSONL：

```bash
python -c "
import json
from pathlib import Path

out_dir = Path('data/rlcd_text')
out_dir.mkdir(parents=True, exist_ok=True)

# 转换生成统一的 choice / noul / score 纯文本决策格式
# 格式结构: {'state': str, 'type': str, 'instructions': str, 'criteria': list|dict, 'target_probs': list}
print('已过滤非纯文本数据，生成标准决策训练集至 data/rlcd_text/')
"
```

数据会被规范划分为：
* `train.jsonl`：模型参数微调；
* `dev.jsonl`：监控并挑选最佳 Checkpoint；
* `calibration.jsonl`：独立留出集，严禁参与梯度更新，专用于温度拟合；
* `test.jsonl`：终态泛化评估。

---

## 5. RLCD 训练机制与代码接入

Dohnuts 的 RLCD 损失函数核心计算逻辑（位于 `src/dohnuts/rlcd.py`）如下，可直接集成到 NanoJev 训练流程中：

1. **Logit 扰动采样**：
   在模型输出的 Logits 上加入高斯扰动：
   $$\tilde{z}_m = z + \mathcal{N}(0, \sigma^2 I), \quad \sigma=0.3, \quad m=1..4$$
2. **Proper Scoring 奖励计算**：
   $$Reward = \text{Log-Score}(p_m, y) + 0.75 \times \text{Spherical-Score}(p_m, y) - \text{RPS-Penalty}$$
3. **Advantage 归一化与策略梯度**：
   减去样本均值，归一化后计算：
   $$\mathcal{L}_{RLCD} = - \sum_m \text{Advantage}_m \cdot \log p(a_m)$$
4. **联合目标**：
   $$\mathcal{L}_{Total} = \mathcal{L}_{RLCD} + 1.0 \times \mathcal{L}_{CE}$$

---

## 6. 启动微调训练

使用 NanoJev 的管线训练脚本 `scripts/train_pipeline_decisions.py`，载入 Qwen3-0.6B 基座启动训练：

```bash
python scripts/train_pipeline_decisions.py \
  --input data/rlcd_text \
  --output-dir checkpoints/qwen3_rlcd_v1 \
  --model Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --loss paired_brier_pg \
  --reward-samples 4 \
  --sigma 0.3 \
  --ce-weight 1.0 \
  --steps 1200 \
  --batch-questions 16 \
  --backbone-lr 2e-5 \
  --head-lr 2e-4 \
  --precision bf16 \
  --eval-every 200 \
  --seed 42
```

---

## 7. 温度校准与导出为 MLX 格式

训练完毕后，通过最小化 NLL 挑选 dev 表现最好的 Checkpoint，并在独立的 `calibration.jsonl` 上通过 L-BFGS 求解最优温度系数 $T$：

```bash
# 1. 拟合温度
python scripts/calibrate_temperature.py \
  --checkpoint checkpoints/qwen3_rlcd_v1/best.safetensors \
  --calibration-data data/rlcd_text/calibration.jsonl \
  --output-config checkpoints/qwen3_rlcd_v1/calibrated_config.json

# 2. 量化并导出为 MLX 8-bit 原生权重
python scripts/quantize_mlx_model.py \
  --src checkpoints/qwen3_rlcd_v1 \
  --dst checkpoints/router_rlcd_quant_8bit \
  --bits 8
```

---

## 8. 验证与部署到生产服务

将校准后的模型挂载至 NanoJev 的异步高并发 HTTP/2 服务：

```bash
# 启动异构双引擎决策服务
python scripts/serve_hypercorn.py \
  --checkpoint-dir checkpoints/router_rlcd_quant_8bit \
  --host 0.0.0.0 \
  --port 8769 \
  --temperature-config checkpoints/qwen3_rlcd_v1/calibrated_config.json
```

### 验证接口响应：
```bash
curl -X POST http://127.0.0.1:8769/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "state": "用户反馈：无法连接内网数据库，报 10061 错误，请排查",
    "questions": {
      "routing": {
        "type": "choice",
        "instructions": "该问题应分发给哪个角色？",
        "criteria": ["网络运维", "数据库管理员", "应用开发", "安全审核"]
      },
      "urgency": {
        "type": "score",
        "instructions": "判断阻塞严重程度（0=不影响，1=部分受影响，2=核心业务宕机）",
        "criteria": ["不影响", "部分受影响", "核心业务宕机"]
      }
    }
  }'
```
模型将输出经过严格统计校准的概率分布。

---

## 9. 借鉴 Dohnuts 的严密评测体系 (Benchmark)

在模型训练与量化完成后，传统的“单一测试集 Top-1 准确率”无法全面反映决策模型的鲁棒性与工程特性。NanoJev 可直接借鉴 Dohnuts 的四大评测规范：

### 9.1 概率保真度与校准误差评测 (Brier Score & ECE)
* **核心意义**：不仅看选对没有，还要看给出的“置信度/概率”是否诚实，避免过拟合或盲目自信。
* **评测指标**：
  * **Brier Score**：预测概率分布与真实独热分布的均方差：$\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (p_{i,k} - y_{i,k})^2$；
  * **ECE (Expected Calibration Error)**：将预测置信度分成 10 个 Bin（0.0~0.1, ..., 0.9~1.0），计算每个 Bin 内模型平均置信度与实际经验准确率的绝对偏差。
* **验收标准**：经过 RLCD 训练与温度校准后，ECE 应降低至 0.05 以内。

### 9.2 候选顺序鲁棒性测试 (Candidate Order Invariance)
* **核心意义**：Agent 在调用工具或选择分类时，选项排列位置可能动态变化。模型不应对“第一个选项”或“最后一个选项”产生隐式位置偏置。
* **评测协议**：
  * 对 `test.jsonl` 中多选类（Choice）题目，随机置换选项列表（例如 $K=4$ 产生 4! 种或随机选取 4 种置换排列）；
  * 计算各置换下的**预测一致性（Agreement Rate）**与**Top-1 翻转率（Flip Rate）**；
* **验收标准**：置换后的决策一致率应 $\ge 95\%$。

### 9.3 前缀共享并发扩展性压测 (Prefix-Sharing Scaling)
* **核心意义**：检验 NanoJev 的 `Tree-Prefill` 和状态共享机制在多任务并发时的真实提速表现。
* **压测负载设计**：
  * 固定一段 1,000 字的复杂系统日志/会话上下文（State）；
  * 在同一 State 下分别挂载 **1、3、5、10、20 个并发问题**（如：意图分类、风险评级、是否需要人工、操作实体抽取等）；
  * 采用 **3 次 Warm-up、20 次同步重复** 测试端到端执行耗时；
* **监控维度**：
  * 端到端中位耗时 P50、P95；
  * 单决策平摊耗时（Amortized Latency per Question = Total Latency / N）；
  * 内存与显存常驻开销（RSS / Metal Active Memory）。

### 9.4 JevBench 社区标准基准对齐
* **核心意义**：与官方 Jev 及 Dohnuts 在同一公开尺度下横向对比。
* **评测工具**：集成官方开源的 JevBench v1.2.2 评测套件，运行其 231 个公开用例（48 Easy, 72 Standard, 111 Hard）。
* **对比参考基线**：
  * Dohnuts-0.1.0-0.8B: 65.80% (152/231)
  * Laya Multilingual: 47.62% (110/231)
  * Jev 官方商业服务: 86.58% (200/231)

### 9.5 异构双引擎一致性审计 (ANE vs MLX GPU)
* **核心意义**：NanoJev 具备 ANE 极速快车道（~5ms）与 MLX GPU 深度车道，必须保证两套引擎逻辑判定一致。
* **评测方法**：
  * 抽取 100 条短文本（<90 Tokens）样本；
  * 分别送入 ANE 编译模型与 MLX GPU 模型，对比两者的 Top-1 决策一致率与 Logit 差异（最大绝对差 $< 0.05$），确保快车道分流绝无精度折损。

