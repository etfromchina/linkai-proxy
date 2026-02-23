# LinkAI Proxy 开发日志

## 2026-02-08

### 问题描述
- `gpt-4o` → `CM5Ex4OE` 正常 (200 OK)
- `gpt-5-mini` → `jPzuwqKZ` 失败 (401 Unauthorized)

### 原因分析
原代码所有模型映射共用同一个 `LINKAI_API_KEY`，但 `jPzuwqKZ` 这个 APP Code 在 LinkAI 后台配置的 API Key 与全局配置不一致，导致认证失败。

### 解决方案
扩展代码支持为不同 APP Code 配置独立的 API Key：

**新增配置项：**
```bash
# APP Code → API Key 映射
APP_CODE_CM5Ex4OE=Link_NTVtgodWFACfMkFhiMVvIiIpayjItqJ61yqiZNcWc8
APP_CODE_jPzuwqKZ=Link_NTVtgodWFACfMkFhiMVvIiIpayjItqJ61yqiZNcWc8
```

**新增代码逻辑：**
```python
# APP Code 到 API Key 映射配置
def parse_app_code_api_keys() -> Dict[str, str]:
    # 解析环境变量中的 APP Code → API Key 映射

def get_api_key_for_app_code(app_code: str) -> str:
    # 根据 APP Code 获取对应的 API Key
    if app_code in APP_CODE_API_KEYS:
        return APP_CODE_API_KEYS[app_code]
    return LINKAI_API_KEY  # 回退到全局 Key
```

### 修改文件
1. `main.py` - 添加多 API Key 支持逻辑
2. `.env` - 添加 `APP_CODE_*` 配置项

### 最终配置
```bash
# 模型 → APP Code 映射
MODEL_MAPPINGS=gpt-4o:CM5Ex4OE,gpt-5-mini:jPzuwqKZ

# APP Code → API Key 映射
APP_CODE_CM5Ex4OE=Link_NTVtgodWFACfMkFhiMVvIiIpayjItqJ61yqiZNcWc8
APP_CODE_jPzuwqKZ=Link_NTVtgodWFACfMkFhiMVvIiIpayjItqJ61yqiZNcWc8
```

### 验证结果
- `gpt-4o` → `CM5Ex4OE` ✅ 200 OK
- `gpt-5-mini` → `jPzuwqKZ` ✅ 200 OK

---

## 历史记录

### 2026-02-07
- 初始化 linkai-proxy 项目
- Docker 容器部署完成
- 基础 OpenAI 兼容 API 代理
- 单模型 (gpt-4o) → 单 APP Code (CM5Ex4OE) 映射
