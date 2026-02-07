"""
LinkAI OpenAI Compatible API Proxy
将 Link-AI API 包装成 OpenAI 格式，支持多模型多 APP Code 映射
"""
import os
import json
import secrets
import re
from typing import Optional, AsyncGenerator, Dict
from starlette.responses import StreamingResponse

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx

load_dotenv()

app = FastAPI(
    title="LinkAI OpenAI Compatible API",
    description="将 Link-AI API 包装成 OpenAI 兼容格式，支持多模型映射",
    version="2.0.0"
)

# ===== 配置 =====
LINKAI_API_KEY = os.getenv("LINKAI_API_KEY")
LINKAI_BASE_URL = os.getenv("LINKAI_BASE_URL", "https://api.link-ai.tech/v1")
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY")
LOCAL_HOST = os.getenv("LOCAL_HOST", "0.0.0.0")
LOCAL_PORT = int(os.getenv("LOCAL_PORT", "8000"))

# ===== 模型到 APP Code 映射配置 =====
def parse_model_app_mapping() -> Dict[str, str]:
    """
    解析环境变量中的模型映射
    环境变量格式: MODEL_MAP_gpt-4o=CM5Ex4OE,MODEL_MAP_gpt-5-mini=jPzuwqKZ
    或者简化的单行格式: MODEL_MAPPINGS=gpt-4o:CM5Ex4OE,gpt-5-mini:jPzuwqKZ
    """
    mappings = {}
    
    # 方式1: 解析 MODEL_MAPPINGS (逗号分隔的 key:value 对)
    model_mappings = os.getenv("MODEL_MAPPINGS", "")
    if model_mappings:
        for pair in model_mappings.split(","):
            if ":" in pair:
                model_id, app_code = pair.split(":", 1)
                mappings[model_id.strip()] = app_code.strip()
    
    # 方式2: 解析单独的 MODEL_MAP_<model>=<app_code> 变量
    for key, value in os.environ.items():
        if key.startswith("MODEL_MAP_"):
            model_id = key[10:]  # 去掉 "MODEL_MAP_" 前缀
            mappings[model_id] = value
    
    return mappings

MODEL_APP_MAPPINGS = parse_model_app_mapping()

# 默认 APP Code (如果没有匹配到模型映射)
DEFAULT_APP_CODE = os.getenv("LINKAI_APP_CODE", "")

# ===== 调试信息 =====
print(f"[LinkAI Proxy] Model→AppCode mappings: {MODEL_APP_MAPPINGS}")
print(f"[LinkAI Proxy] Default APP Code: {DEFAULT_APP_CODE}")


def get_app_code_for_model(model_id: Optional[str]) -> str:
    """
    根据模型ID获取对应的 APP Code
    支持模糊匹配和前缀匹配
    """
    if not model_id:
        return DEFAULT_APP_CODE
    
    # 精确匹配
    if model_id in MODEL_APP_MAPPINGS:
        return MODEL_APP_MAPPINGS[model_id]
    
    # 前缀匹配 (例如: "gpt-4o" 匹配 "gpt-4o-*" )
    for mapping_model, app_code in MODEL_APP_MAPPINGS.items():
        if model_id.startswith(mapping_model):
            return app_code
    
    # 后缀匹配 (例如: "gpt-4o" 匹配 "*-gpt-4o" )
    for mapping_model, app_code in MODEL_APP_MAPPINGS.items():
        if model_id.endswith(mapping_model):
            return app_code
    
    # 包含匹配 (例如: "gpt-4o" 匹配 "*gpt-4o*" )
    for mapping_model, app_code in MODEL_APP_MAPPINGS.items():
        if mapping_model in model_id:
            return app_code
    
    # 没有匹配到，返回默认
    return DEFAULT_APP_CODE


# ===== 鉴权 =====
security = APIKeyHeader(name="Authorization", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Depends(security)) -> str:
    """验证 API Key"""
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # 去掉 "Bearer " 前缀
    if api_key.startswith("Bearer "):
        api_key = api_key[7:]
    
    if api_key != LOCAL_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return api_key


# ===== Pydantic Models =====
from typing import Union

class Message(BaseModel):
    role: str
    content: Union[str, list]


class ChatRequest(BaseModel):
    messages: list[Message]
    model: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    stop: Optional[list[str]] = None


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: Optional[str] = None
    object: str = "chat.completion"
    created: int
    model: Optional[str] = None
    choices: list[Choice]
    usage: Optional[Usage] = None


# ===== 路由 =====
@app.get("/v1/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    """返回模型列表（包含所有映射的模型）"""
    models = []
    
    # 添加映射的模型
    for model_id in MODEL_APP_MAPPINGS.keys():
        models.append({
            "id": model_id,
            "object": "model",
            "created": 0,
            "owned_by": "linkai"
        })
    
    # 添加默认模型
    if DEFAULT_APP_CODE:
        models.append({
            "id": "linkai-default",
            "object": "model",
            "created": 0,
            "owned_by": "linkai"
        })
    
    return {
        "object": "list",
        "data": models
    }


@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    """创建对话（OpenAI 兼容格式）"""
    
    # 根据请求的模型获取对应的 APP Code
    app_code = get_app_code_for_model(request.model)
    
    if not app_code:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' not found in mappings and no default APP Code configured"
        )
    
    print(f"[LinkAI Proxy] Model: {request.model} → APP Code: {app_code}")
    
    # 构建 Link-AI 请求（支持多模态消息）
    linkai_messages = []
    for msg in request.messages:
        linkai_messages.append({
            "role": msg.role,
            "content": msg.content
        })
    
    linkai_body = {
        "app_code": app_code,
        "messages": linkai_messages
    }
    
    # 如果请求指定了模型，也传给 Link-AI
    # if request.model:
    #     linkai_body["model"] = request.model
    
    if request.temperature is not None:
        linkai_body["temperature"] = request.temperature
    if request.top_p is not None:
        linkai_body["top_p"] = request.top_p
    
    # 流式输出
    if request.stream:
        generator = linkai_stream_generator(linkai_body, request.model or "unknown")
        return StreamingResponse(
            StreamingResponseWrapper(generator),
            media_type="text/event-stream"
        )
    
    # 非流式输出
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{LINKAI_BASE_URL}/chat/completions",
            json=linkai_body,
            headers={"Authorization": f"Bearer {LINKAI_API_KEY}"},
            timeout=60.0
        )
        
        if response.status_code != 200:
            error_data = response.json().get("error", {})
            raise HTTPException(
                status_code=response.status_code,
                detail=error_data.get("message", "LinkAI API Error")
            )
        
        linkai_response = response.json()
        
        # 转换为 OpenAI 格式
        choices = linkai_response.get("choices", [])
        openai_choices = []
        
        for i, choice in enumerate(choices):
            openai_choices.append({
                "index": i,
                "message": {
                    "role": choice.get("message", {}).get("role", "assistant"),
                    "content": choice.get("message", {}).get("content", "")
                },
                "finish_reason": choice.get("finish_reason") or "stop"
            })
        
        usage = linkai_response.get("usage", {})
        
        return {
            "id": f"chatcmpl-{secrets.token_hex(8)}",
            "object": "chat.completion",
            "created": int(__import__("time").time()),
            "model": request.model or "linkai-default",
            "choices": openai_choices,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            }
        }


class StreamingResponseWrapper:
    """包装异步生成器为可迭代对象"""
    
    def __init__(self, generator):
        self._generator = generator
    
    def __aiter__(self):
        return self._generator
    
    async def __anext__(self):
        try:
            return await self._generator.__anext__()
        except StopAsyncIteration:
            raise StopAsyncIteration


async def linkai_stream_generator(linkai_body: dict, model_id: str):
    """异步生成器用于流式响应"""
    linkai_body["stream"] = True
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            f"{LINKAI_BASE_URL}/chat/completions",
            json=linkai_body,
            headers={"Authorization": f"Bearer {LINKAI_API_KEY}"},
            timeout=120.0
        ) as response:
            
            if response.status_code != 200:
                error_msg = "LinkAI API Error"
                try:
                    error_data = await response.json()
                    error_msg = error_data.get("error", {}).get("message", error_msg)
                except:
                    pass
                
                yield f'data: ' + json.dumps({"error": {"message": error_msg}}) + '\n\n'
                yield "data: [DONE]\n\n"
                return
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break
                    
                    try:
                        linkai_data = json.loads(data)
                        
                        # 转换为 OpenAI 格式
                        choices = linkai_data.get("choices", [])
                        openai_choices = []
                        
                        for i, choice in enumerate(choices):
                            delta = choice.get("delta", {})
                            finish_reason = choice.get("finish_reason")
                            
                            openai_choices.append({
                                "index": i,
                                "delta": {"content": delta.get("content", "")},
                                "finish_reason": finish_reason
                            })
                        
                        openai_data = {
                            "id": f"chatcmpl-{secrets.token_hex(8)}",
                            "object": "chat.completion.chunk",
                            "created": int(__import__("time").time()),
                            "model": model_id,
                            "choices": openai_choices
                        }
                        
                        # 如果有 usage 信息（在最后一个 chunk）
                        usage = linkai_data.get("usage") or choice.get("usage", {})
                        if usage and finish_reason:
                            openai_data["usage"] = {
                                "prompt_tokens": usage.get("prompt_tokens", 0),
                                "completion_tokens": usage.get("completion_tokens", 0),
                                "total_tokens": usage.get("total_tokens", 0)
                            }
                        
                        yield f"data: {json.dumps(openai_data)}\n\n"
                        
                    except json.JSONDecodeError:
                        continue


class LinkAIStreamingResponse:
    """流式响应（兼容旧版本）- 已废弃，使用 linkai_stream_generator"""
    
    def __init__(self, linkai_body: dict, api_key: str, model_id: str = "unknown"):
        self.linkai_body = linkai_body
        self.linkai_body["stream"] = True
        self.api_key = api_key
        self.model_id = model_id
    
    async def __call__(self, request: Request):
        async for chunk in linkai_stream_generator(self.linkai_body, self.model_id):
            yield chunk


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "version": "2.0.0",
        "mappings_count": len(MODEL_APP_MAPPINGS)
    }


@app.get("/")
async def root():
    """API 信息"""
    return {
        "name": "LinkAI OpenAI Compatible API",
        "version": "2.0.0",
        "description": "支持多模型多 APP Code 映射",
        "endpoints": {
            "chat_completions": "POST /v1/chat/completions",
            "models": "GET /v1/models",
            "health": "GET /health"
        },
        "model_mappings": MODEL_APP_MAPPINGS,
        "default_app_code": DEFAULT_APP_CODE
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=LOCAL_HOST, port=LOCAL_PORT)
