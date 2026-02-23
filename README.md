# LinkAI OpenAI 兼容代理 (v2.2)

将 Link-AI API 包装成 OpenAI 格式，支持多模型多 APP Code 映射，并支持内存类 Prompt 自动路由。

## 新功能 (v2.2)

- ✅ 多模型映射：不同模型对应不同 APP Code
- ✅ 智能匹配：支持精确匹配、前缀匹配、包含匹配
- ✅ 内存路由：自动识别两类内存任务 Prompt 并切到指定 APP Code
- ✅ 向后兼容：未命中内存路由时继续按 `model -> app_code` 逻辑

## 快速开始

### 1. 配置环境变量

编辑 `.env` 文件：

```bash
# LinkAI API 配置
LINKAI_API_KEY=你的LinkAI_API_KEY
LINKAI_BASE_URL=https://api.link-ai.tech/v1

# 默认 APP Code（当模型未匹配时使用）
LINKAI_APP_CODE=默认AppCode

# 模型映射（普通对话）
MODEL_MAPPINGS=gpt-4o:CM5Ex4OE,gpt-5-mini:jPzuwqKZ

# 可选：不同 APP Code 使用不同 LinkAI API Key
# APP_CODE_CM5Ex4OE=key_for_cm5
# APP_CODE_jPzuwqKZ=key_for_jpz

# 内存 Prompt 自动路由（推荐开启）
MEMORY_ROUTING_ENABLED=true
MEMORY_ROUTING_MIN_SCORE=2
MEMORY_ROUTING_DEBUG=false
MEMORY_MANAGER_APP_CODE=VBLtq5no
MEMORY_EXTRACTOR_APP_CODE=VBLtq5no

# 本地访问鉴权
LOCAL_API_KEY=sk-your-local-key
LOCAL_HOST=0.0.0.0
LOCAL_PORT=8000
```

### 2. 启动服务

```bash
docker compose up -d --build
```

## API 使用

### 基础信息

- Base URL: `http://localhost:8000/v1`
- API Key: `LOCAL_API_KEY`

### 对话接口

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-local-key" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

## 映射配置说明

### 模型映射格式

格式1：单行逗号分隔

```bash
MODEL_MAPPINGS=gpt-4o:CM5Ex4OE,gpt-5-mini:jPzuwqKZ,claude-3:ABC123
```

格式2：独立环境变量

```bash
MODEL_MAP_gpt-4o=CM5Ex4OE
MODEL_MAP_gpt-5-mini=jPzuwqKZ
MODEL_MAP_claude-3=ABC123
```

### 普通请求匹配优先级

1. 精确匹配：`gpt-4o` -> `CM5Ex4OE`
2. 前缀匹配：`gpt-4o-preview` -> 匹配 `gpt-4o`
3. 后缀/包含匹配：`xxx-gpt-4o` 或 `gpt-4o-2024`
4. 默认：`LINKAI_APP_CODE`

## 内存路由说明（新增）

### 目标

解决某些 Agent 平台会把“内存管理/事实抽取”Prompt 与普通对话一起发到同一模型的问题，避免普通 Agent 工作流被内存任务干扰。

### 路由规则

当 `MEMORY_ROUTING_ENABLED=true` 时：

1. 命中 **Memory Manager** 特征词（如 `You are a smart memory manager`、`old_memory`、`ADD/UPDATE/DELETE/NONE`）
: 路由到 `MEMORY_MANAGER_APP_CODE`
2. 命中 **Personal Information Organizer** 特征词（如 `You are a Personal Information Organizer`、`Output facts`）
: 路由到 `MEMORY_EXTRACTOR_APP_CODE`
3. 未命中
: 走原有 `model -> app_code` 映射

### 路由优先级

`memory_manager` > `memory_extractor` > `default`

### 调试

设置 `MEMORY_ROUTING_DEBUG=true` 可输出命中分数与路由标签（不会打印完整敏感密钥）。

## API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/v1/chat/completions` | POST | 创建对话 |
| `/v1/models` | GET | 获取模型列表 |
| `/health` | GET | 健康检查 |
| `/` | GET | API 信息 |

## 故障排查

1. 400 错误：模型未找到映射且无默认 APP Code
2. 401 错误：检查 `Authorization: Bearer <LOCAL_API_KEY>`
3. 内存任务未切换：检查 `MEMORY_ROUTING_ENABLED` 与 `MEMORY_*_APP_CODE`
4. 容器 `unhealthy`：检查 `docker compose ps` 与 `/health`

## 停止服务

```bash
docker compose down
```
