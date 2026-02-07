# LinkAI OpenAI 兼容代理 (v2.0)

将 Link-AI API 包装成 OpenAI 格式，支持多模型多 APP Code 映射。

## 新功能 (v2.0)

- ✅ **多模型映射**: 不同模型对应不同的 APP Code
- ✅ **智能匹配**: 支持精确匹配、前缀匹配、包含匹配
- ✅ **向后兼容**: 未匹配的模型使用默认 APP Code

## 快速开始

### 1. 配置环境变量

编辑 `.env` 文件：

```bash
# LinkAI API 配置
LINKAI_API_KEY=你的LinkAI_API_KEY
LINKAI_BASE_URL=https://api.link-ai.tech/v1

# 默认 APP Code（可选）
LINKAI_APP_CODE=默认AppCode

# 模型映射配置
MODEL_MAPPINGS=gpt-4o:CM5Ex4OE,gpt-5-mini:jPzuwqKZ
```

### 2. 启动服务

```bash
docker-compose up -d
```

## API 使用

### 基础信息

- **Base URL**: `http://localhost:8000/v1`
- **API Key**: `sk-05bd07450617f210e1c5e6b3d45972fa`

### 对话接口

```bash
# 调用 gpt-4o (自动映射到 CM5Ex4OE)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-05bd07450617f210e1c5e6b3d45972fa" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "你好"}]
  }'

# 调用 gpt-5-mini (自动映射到 jPzuwqKZ)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-05bd07450617f210e1c5e6b3d45972fa" \
  -d '{
    "model": "gpt-5-mini",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

### Python 调用示例

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-05bd07450617f210e1c5e6b3d45972fa"
)

# 调用 gpt-4o
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "请介绍自己"}]
)
print(response.choices[0].message.content)

# 调用 gpt-5-mini
response = client.chat.completions.create(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "请介绍自己"}]
)
```

## 映射配置说明

### 格式1: 单行逗号分隔

```bash
MODEL_MAPPINGS=gpt-4o:CM5Ex4OE,gpt-5-mini:jPzuwqKZ,claude-3:ABC123
```

### 格式2: 独立环境变量

```bash
MODEL_MAP_gpt-4o=CM5Ex4OE
MODEL_MAP_gpt-5-mini=jPzuwqKZ
MODEL_MAP_claude-3=ABC123
```

### 匹配优先级

1. **精确匹配**: `gpt-4o` → `CM5Ex4OE`
2. **前缀匹配**: `gpt-4o-preview` → 匹配 `gpt-4o` → `CM5Ex4OE`
3. **包含匹配**: `gpt-4o-2024` → 包含 `gpt-4o` → `CM5Ex4OE`
4. **默认**: 使用 `LINKAI_APP_CODE` 或报错

### 示例配置

```bash
# 完整示例
LINKAI_API_KEY=sk-xxx
LINKAI_APP_CODE=DEFAULT_CODE
MODEL_MAPPINGS=gpt-4o:APP_CODE_1,gpt-5-mini:APP_CODE_2,claude-3-opus:APP_CODE_3
```

## API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/v1/chat/completions` | POST | 创建对话 |
| `/v1/models` | GET | 获取模型列表 |
| `/health` | GET | 健康检查 |
| `/` | GET | API 信息 |

## 在 OpenClaw 中使用

```yaml
openai:
  api_key: "sk-05bd07450617f210e1c5e6b3d45972fa"
  base_url: "http://localhost:8000/v1"
```

## 故障排查

1. **400 错误**: 模型未找到映射且无默认 APP Code
2. **401 错误**: 检查 API Key 是否正确
3. **连接失败**: 确认 Docker 容器已启动

## 停止服务

```bash
docker-compose down
```
