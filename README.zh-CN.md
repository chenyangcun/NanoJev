# NanoJev-MLX — 苹果芯片原生并行决策模型 (Apple Silicon Native)

[English](README.md) · **简体中文**

**基于 Qwen3-0.6B 骨干网络的 Apple Silicon 原生并行决策模型。输入应用状态与结构化问题，单次前向输出完整概率分布——零自回归 Token 解码。**

NanoJev-MLX Fork 并扩展自上游开源项目 [TianyuCodings/NanoJev](https://github.com/TianyuCodings/NanoJev)，复刻并深入推进了 [TypeSafe Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 的 System One 决策范式，打造了一套完全本地化、高并发、超低功耗的智能决策引擎。结合 Apple MLX 与 Core ML ANE，实现了 **Neural Engine (NPU) ~5ms 极速响应** 与 **8-bit Metal GPU ~230ms 复杂多轮深度推理**。

[上游源项目 (TianyuCodings/NanoJev)](https://github.com/TianyuCodings/NanoJev) · [架构调优全景报告 (技术总结)](docs/OPTIMIZATION_SUMMARY.md) · [TypeSafe API 协议规范](docs/TYPESAFE_CONTRACT.md) · [历史基准测评](docs/DEVELOPMENT_RESULTS.md)

---

## ⚡ 核心架构与技术创新

- **异构双引擎并发路由 (`DualEngineRouter`)**：
  - **ANE 极速快车道 (`~5ms P50`)**：针对小于 90 Tokens 的微型命题、轻量选择与布尔判断，自动分流至苹果专属 NPU（Apple Neural Engine），零占用 GPU 显存，单次决策功耗低至 ~0.15 焦耳。
  - **Metal GPU 深度主车道 (`~250ms`)**：针对包含代码变更、工具调用日志、多轮长会话的复杂任务，分流至基于 MLX 的 8-bit 量化 Qwen3-0.6B 骨干网深度推理。
- **跨问题三级树状状态共享 (Tree-Prefill State Sharing)**：
  - 无论单次请求挂载多少个问题，千字长文本 `State` 整个请求全局**仅前向预填充 1 次**；
  - 候选分支通过零拷贝批次广播并行叉分计算，实际 Token 计算量**暴降 84.5%**。
- **静态原地复用 KV-Cache 内存池 (`StaticKVCachePool`)**：
  - 全局静态提示词缓存，每次请求完成后通过指针原地清零重置（`.trim(offset)`），实现**零动态内存申请、零垃圾回收（GC）卡顿**。
- **MLX 原生 8-bit Affine 矩阵量化**：
  - 骨干线性层从 2.27GB 深度压缩至 **1.06GB（体积缩减 53.0%）**，可在有限统一内存环境下与其他 30B+ 大模型平稳共存不爆显存。
- **自适应多特征置信度工程 (`Adaptive Multi-Feature Confidence`)**：
  - 融合 Top-1 优势度、候选分离边距（$\Delta = \text{Top1} - \text{Top2}$）、归一化香农熵余数与容量因子，结合动态自适应温度调节（$T = 0.25 \sim 0.85$）；
  - 模糊任务平滑输出不确定信号，明确任务置信度强力突破至 **0.99 ~ 1.00**。
- **模块化可插拔多头容器 (`MultiHeadRegistry`)**：
  - 解耦骨干网与任务分类头，支持按业务域独立挂载 `router`（代码路由）、`skill`（技能选择）、`agent`（分派调度）和 `news`（新闻价值分析）专属头；
  - 支持外挂加载独立轻量权重文件（每个 Head 仅约 2MB），并内置启发式路由审计日志自动追加记录。
- **生产级 HTTP/2 & HTTP/1.1 ASGI 异步服务**：
  - 基于 Hypercorn 原生支持 HTTP/2 二进制帧多路复用与 120 秒 Keep-Alive 连接池，单机并发吞吐高达 **91.9 req/s**；
  - 100% 兼容 TypeSafe 官方 `POST /v1/systemone` 及 NanoJev 原生 `POST /api/evaluate`。
- **双端 Shadow 对齐与增量自进化闭环 (Data Flywheel)**：
  - 本机 Crontab 每小时增量运行 (`scripts/daily_shadow_auto_loop.sh`)，基于 `request_id` 自动对齐路由器生产 Shadow 日志与云端官方 Jev 结果；
  - 自动捕获分歧样本并转化为标准微调数据集，满额自动触发后台静默训练与平滑热重载。

---

## 📊 评估测试对比总览

基于 36 场景中英双语基准与 24 场景真实会话压缩状态基准实测验证：

| 评估维度 | 初始基线 (Baseline) | 当前版本 (NanoJev-MLX) | 达标状态 |
| :--- | :---: | :---: | :---: |
| **真实多轮会话批测 (24 场景)** | 10 / 24 (41.7%) ❌ | **24 / 24 (100.0%)** 🌟 | **100% 满分命中** |
| **中英全量基础场景 (36 场景)** | 2 / 12 (16.7%) ❌ | **35 / 36 (97.2%)** 🌟 | **生产就绪** |
| **高风险生产操作拦截召回率 (`>=0.5`)** | 0 / 6 (0.0%) ❌ 严重漏检 | **100.0% (17/17 全拦截)** 🚨 | **高危零漏检** |
| **低风险操作虚警误报率** | 7 个误报 (虚警率 41%) ❌ | **0 误报 (0.0%)** ✅ | **日常零噪音** |
| **决策置信度表现 (Confidence)** | 0.18 ~ 0.23 (过低导致降级) | **绝大部分达 0.98 ~ 1.00** ⚡ | **果断清晰** |
| **显存物理占用空间** | 2.27 GB (容易引起 OOM) | **1.06 GB (显存直降 53%)** | **极致轻量** |
| **端到端中位耗时 (P50)** | ~870 ms (长尾达 4.7s) | **~258 ms (长尾压入 380ms 内)** | **提速 3.3 倍** |

---

## 🚀 苹果电脑 (macOS) 快速上手

### 1. 准备环境

运行要求：macOS 14+（推荐 macOS 15+）、Apple Silicon 芯片（M1/M2/M3/M4）、Python 3.11+。

```bash
git clone git@github.com:chenyangcun/NanoJev-MLX.git
cd NanoJev-MLX

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-mlx.txt
```

### 2. 下载预训练权重

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='C-Tianyu/NanoJev', local_dir='checkpoints/NanoJev')
"
```

### 3. 启动高并发 HTTP/2 决策服务

```bash
# 启动异构双引擎 (ANE 极速快车道 + 8-bit MLX GPU) 服务，监听 8769 端口
python3 scripts/serve_hypercorn.py \
  --checkpoint-dir checkpoints/router_quant_8bit \
  --host 0.0.0.0 \
  --port 8769 \
  --temperature 0.35 \
  --max-length 4096
```

健康检查：
```bash
curl -s http://127.0.0.1:8769/api/health
```

---

## 📡 API 调用示例 (兼容 TypeSafe System One)

NanoJev-MLX 提供完全兼容官方标准的 `POST /v1/systemone` 端点：

```bash
curl -X POST http://127.0.0.1:8769/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jev-latest",
    "state": {
      "user_task": "审查 OAuth 回调流程是否存在令牌泄露风险，提出代码修改建议，但不要部署。"
    },
    "questions": {
      "complexity": {
        "type": "choice",
        "instructions": "判断接下来 Coding Agent 调用的任务复杂度。",
        "criteria": {
          "bounded": "单文件局部微调或改错",
          "standard": "日常功能需求开发",
          "complex": "跨模块架构迁移、并发竞态排查或安全审查",
          "exceptional": "生产严重事故干预或重大故障"
        }
      },
      "high_risk": {
        "type": "noul",
        "instructions": "该任务是否涉及生产破坏或高危操作？"
      }
    }
  }'
```

**响应报文**：
```json
{
  "model": "jev-latest",
  "answers": {
    "complexity": {
      "type": "choice",
      "choice": "complex",
      "probabilities": {
        "bounded": 0.0,
        "standard": 0.0,
        "complex": 1.0,
        "exceptional": 0.0
      },
      "confidence": 1.0
    },
    "high_risk": {
      "type": "noul",
      "noul": 0.9998
    }
  },
  "usage": {
    "input_tokens": 573,
    "output_tokens": 0
  }
}
```

---

## 🛠️ CLI 常用管理命令

### 本地开机自启守护进程管理 (LaunchAgent)
在 M1 Mac Studio 或服务端：
```bash
nanojev-service status   # 查看服务运行状态、PID 与健康指标
nanojev-service restart  # 重启常驻服务
nanojev-service logs     # 查看实时推理与访问日志
```

### 基准评测与压测
```bash
# 执行完整真实压缩会话场景评测 (24 场景)
python3 /path/to/evaluate-local-jev.py --cases /tmp/jev-realistic-cases.json

# HTTP/2 与 HTTP/1.1 延迟与高并发吞吐基准压测
python3 scripts/benchmark_http2.py
python3 scripts/benchmark_concurrency.py

# 单元测试 (多头注册表与分发规则验证)
python3 scripts/test_multi_head_registry.py
```

### 模型量化与结构剪枝
```bash
# 导出 8-bit MLX 量化模型 (显存直降 53%)
python3 scripts/quantize_mlx_model.py \
  --source-dir checkpoints/router_realistic_mlp \
  --target-dir checkpoints/router_quant_8bit \
  --bits 8

# 结构化层剪枝 (如 28 层剪枝为 14 层)
python3 scripts/prune_mlx_model.py \
  --source-dir checkpoints/router_realistic_mlp \
  --target-dir checkpoints/router_pruned_14l \
  --target-layers 14
```

---

## 📂 核心代码目录结构

```text
NanoJev-MLX/
├── checkpoints/                 # 模型检查点目录 (8-bit量化权重、可插拔头)
├── data/                        # 对齐训练集与 Shadow 自动蒸馏数据集
├── docs/
│   ├── OPTIMIZATION_SUMMARY.md  # 15 项深度技术调优全景总结报告
│   ├── TYPESAFE_CONTRACT.md     # TypeSafe System One 兼容性契约与输入规范
│   └── DEVELOPMENT_RESULTS.md   # 初始游戏与导航基准开发记录
├── scripts/
│   ├── dual_engine_router.py    # ANE (NPU) + MLX (GPU) 异构并发分流路由器
│   ├── cross_question_sharing_engine.py  # 三级树状跨问题状态共享与早停引擎
│   ├── fast_decision_engine.py  # 静态就地复用 KV-Cache 内存池与 JIT 内核融合
│   ├── mlx_multi_head_registry.py # 模块化可插拔多头容器与启发式审计日志
│   ├── adaptive_confidence.py   # 自适应多信号置信度特征工程算法
│   ├── serve_hypercorn.py       # 高并发 HTTP/2 + HTTP/1.1 异步 ASGI 服务入口
│   ├── asgi_app.py              # 异步 ASGI 端点路由与数据适配
│   ├── quantize_mlx_model.py    # 原生 8-bit/4-bit 权重量化脚本
│   ├── shadow_pipeline.py       # 双端日志自动配对分析与分歧样本萃取
│   └── daily_shadow_auto_loop.sh # 每小时定时增量自进化闭环脚本
└── requirements-mlx.txt         # 纯净 Apple Silicon 原生运行依赖清单
```

---

## 📄 开源许可与致谢

- 核心实现基于 **Apache-2.0** 许可证开源。
- 本项目 Fork 并演进自 Tianyu Chen 开源的原始项目 [TianyuCodings/NanoJev](https://github.com/TianyuCodings/NanoJev)。
- 骨干网络源自阿里巴巴通义千问团队开源的 **Qwen3-0.6B** 模型。
- 决策接口定义与 System One 范式受 [TypeSafe AI](https://typesafe.ai) 启发。
- ANE 与 Core ML 部分设计参考了 [Laya](https://github.com/NandhaKishorM/laya) 与 [laya-coreml](https://github.com/mizorewww/laya-coreml)。
