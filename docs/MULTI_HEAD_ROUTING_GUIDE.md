# NanoJev 多判决头调度与接入使用指南 (Multi-Head Routing Guide)

本文档说明使用方（如 `jev-cliproxy-router`、各类 Agent 网关或自定义客户端）如何与 NanoJev 的 **模块化可插拔多头架构（`MultiHeadRegistry`）** 进行对接与参数配置。

---

## 一、架构背景与优势

在传统模型架构中，所有分类/判断任务都由单一线性层完成，极易导致“复杂度偏好”与“技能选择偏好”相互干扰（例如把代码量大的任务误判为必须选某个技能）。

NanoJev 采用了 **“共享一套 0.6B 骨干网 + 业务专属分类头（各仅 ~2MB）”** 的设计：
- **`router` 头**：专门负责任务复杂度（`complexity`）、高风险操作拦截（`high_risk`）、独立执行性（`independent`）；
- **`skill` 头**：专门负责技能初筛（`shortlist`）与技能可用性二次校验（`verify_<skill_id>`）；
- **`agent` 头**（预留）：负责 Subagent 派发路由（如 Codex 内部执行 vs OpenCode 深度委托）；
- **`news` 头**（预留）：负责新闻价值、情感与主题分类。

---

## 二、接入与传参方式

NanoJev 提供 **零侵入自动启发式** 与 **显式精准控制** 两种接入方式：

### 方式 1：全自动模式（Zero-Config，推荐 ⭐⭐⭐⭐⭐）

使用方**完全不需要修改任何现有调用代码或参数**，保持原样发送请求即可。

引擎内置的启发式路由器会根据问题的 `qid` 和指令内容自动指派：

| 匹配规则 | 自动指派的 Head | 典型 QID / 提问场景 |
| :--- | :--- | :--- |
| 包含 `complex`、`risk`、`indep`、`route`、`tier` | **`router`** | `complexity`, `high_risk`, `independent`, `route_tier` |
| 包含 `shortlist`、`skill`、`tool`、`plugin` 或以 `verify_` 开头 | **`skill`** | `shortlist`, `verify_drawio_skill`, `verify_chrome_cdp`, `tool_choice` |
| 包含 `agent`、`subagent`、`worker`、`delegat` | **`agent`** | `agent_selection`, `subagent_dispatch` |
| 包含 `news`、`editorial`、`sentiment` | **`news`** | `news_value`, `content_quality` |
| 未命中以上规则 | **`default` (即 `router`)** | 安全回退到默认头 |

> **提示**：触发自动启发式或回退时，服务端会自动异步追加一条 JSON 审计日志至 `~/Library/Logs/NanoJev/head_dispatch.jsonl`，方便后续分析和微调收集。

---

### 方式 2：显式参数指定（Explicit Control，精准控制 ⭐⭐⭐⭐）

若使用方新增了自定义命名的提问 ID，希望完全跳过规则推断、强制由指定头处理，只需在对应的 `question` 对象中添加 **`"head"`** 字段。

#### 1. 参数定义

| 参数名 | 字段类型 | 是否必填 | 可选值 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| **`head`** | `string` | 可选 | **`"router"`** | 任务复杂度、高危判定、独立执行性 |
| | | | **`"skill"`** | 技能推荐与可用性验证 |
| | | | **`"agent"`** | 智能体 / Subagent 分发决策 |
| | | | **`"news"`** | 新闻价值与分类决策 |
| | | | **`"default"`** | 默认通用头 |

#### 2. 请求报文示例（混合调用不同 Head）

在同一个 HTTP 请求中，不同问题可以灵活指向不同的 Head，且底层**完全共享同一个任务上下文的 State 前缀计算**：

```json
{
  "model": "jev-latest",
  "state": {
    "user_task": "帮我排查订单支付链路的并发死锁，并画一张微服务架构图"
  },
  "questions": {
    "task_complexity": {
      "head": "router",
      "type": "choice",
      "instructions": "判断任务复杂度",
      "criteria": {
        "bounded": "单文件局部微调",
        "standard": "日常功能需求开发",
        "complex": "跨模块核心重构或复杂竞态排查",
        "exceptional": "生产严重故障干预"
      }
    },
    "selected_skill": {
      "head": "skill",
      "type": "choice",
      "instructions": "选择最匹配该任务的技能",
      "criteria": {
        "none": "不需要额外技能",
        "drawio-skill": "生成可继续编辑的 Draw.io 图表",
        "chrome-cdp": "操作已打开的 Chrome 浏览器"
      }
    },
    "risk_check": {
      "head": "router",
      "type": "noul",
      "instructions": "该任务是否涉及生产数据破坏或高危操作？"
    }
  }
}
```

---

## 三、响应结果解析与调试元数据

调用返回时，每个答案除了给出常规的 `choice` / `noul`、`probabilities` 和 `confidence` 之外，还会附带调度的跟踪元数据：

```json
{
  "model": "jev-latest",
  "answers": {
    "task_complexity": {
      "type": "choice",
      "choice": "complex",
      "probabilities": {
        "bounded": 0.0,
        "standard": 0.0,
        "complex": 1.0,
        "exceptional": 0.0
      },
      "confidence": 1.0,
      "used_head": "router",
      "routing_mode": "explicit"
    },
    "selected_skill": {
      "type": "choice",
      "choice": "drawio-skill",
      "probabilities": {
        "none": 0.0,
        "drawio-skill": 1.0,
        "chrome-cdp": 0.0
      },
      "confidence": 1.0,
      "used_head": "skill",
      "routing_mode": "explicit"
    },
    "risk_check": {
      "type": "noul",
      "noul": 0.0001,
      "used_head": "router",
      "routing_mode": "explicit"
    }
  },
  "usage": {
    "input_tokens": 624,
    "output_tokens": 0
  }
}
```

### 元数据字段释义：
- **`used_head`**：本次判断实际执行的 Head 名称（如 `"router"`, `"skill"`）；
- **`routing_mode`**：
  - `"explicit"`：由客户端在请求中显式传参指定；
  - `"heuristic:pattern:..."`：通过关键词模式自动推断指派；
  - `"fallback:default"`：未命中任何规则，由默认兜底头处理。

---

## 四、Python 与 cURL 快速测试示例

### 1. cURL 调用（零配置自动分发）
```bash
curl -X POST http://192.168.123.88:8769/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jev-latest",
    "state": { "user_task": "请把系统模块关系画成可继续编辑的 Draw.io 架构图。" },
    "questions": {
      "shortlist": {
        "type": "choice",
        "instructions": "选择最相关的技能",
        "criteria": {
          "none": "不需要技能",
          "drawio-skill": "创建可编辑的 drawio 架构图",
          "video-parse": "音视频字幕提取"
        }
      }
    }
  }'
```

### 2. Python 显式控制调用
```python
import requests

url = "http://192.168.123.88:8769/v1/systemone"

payload = {
    "model": "jev-latest",
    "state": {"user_task": "提取这个 Bilibili 视频的字幕"},
    "questions": {
        "pick_skill": {
            "head": "skill",  # 显式指定走技能头
            "type": "choice",
            "instructions": "选择该任务需要注入的 Skill",
            "criteria": {
                "none": "无专属技能",
                "video-parse": "B站/YouTube视频解析与字幕转写",
                "opencli": "网页抓取与网站适配器",
            },
        }
    },
}

resp = requests.post(url, json=payload).json()
ans = resp["answers"]["pick_skill"]
print(f"推荐技能: {ans['choice']} (置信度: {ans['confidence']}, 执行头: {ans['used_head']})")
```
